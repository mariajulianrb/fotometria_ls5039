import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

# ====================================================================
# 1. Arquivos e Configurações (Filtro B via Gaia DR3)
# ====================================================================
INPUT_CSV_TOTAL = 'fotometria_final_B.csv'
INPUT_CSV_FILTRADO = 'fotometria_limpa_B.csv'
OUTPUT_CSV = 'fotometria_B_calibrada_Gaia.csv'

RAIO_MATCH_ARCSEC = 4.0 
ZP_OFFSET = 25.0

def ajuste_linear(x, alfa, c):
    return alfa * x + c

def transformar_gaia_para_B(G, BP, RP):
    """Transformação empírica de Gaia (G, BP, RP) para a banda B clássica"""
    bp_rp = BP - RP
    B_minus_G = 0.0176 + 0.4186 * bp_rp + 0.0881 * (bp_rp**2) - 0.0125 * (bp_rp**3)
    return G + B_minus_G

# ====================================================================
# 2. Leitura e Preparação dos Dados Instrumentais
# ====================================================================
df_total = pd.read_csv(INPUT_CSV_TOTAL)
df_filt = pd.read_csv(INPUT_CSV_FILTRADO)

coluna_erro = 'Erro_Mag' if 'Erro_Mag' in df_total.columns else 'Erro_Mag_Inst'

std_fundo = df_total['Std_Fundo_Local'].iloc[0] if 'Std_Fundo_Local' in df_total.columns else 10.0
area_ap = df_total['Area_Ap'].iloc[0] if 'Area_Ap' in df_total.columns else 60.0
exptime = df_total['Exptime'].iloc[0] if 'Exptime' in df_total.columns else 1.0

# Aplica o Zero Point
df_total['Mag_Inst'] = df_total['Mag_Inst'] + ZP_OFFSET
df_filt['Mag_Inst']  = df_filt['Mag_Inst'] + ZP_OFFSET

coords_total = SkyCoord(ra=df_total['RA_deg'].values * u.deg, dec=df_total['Dec_deg'].values * u.deg)
coords_filt  = SkyCoord(ra=df_filt['RA_deg'].values * u.deg,  dec=df_filt['Dec_deg'].values * u.deg)

# ====================================================================
# 3. Consulta ao Gaia DR3 (Baixando G, BP e RP)
# ====================================================================
centro = SkyCoord(ra=df_total['RA_deg'].mean() * u.deg, dec=df_total['Dec_deg'].mean() * u.deg)
vizier = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag', 'BPmag', 'RPmag'], row_limit=-1)

# Raio amplo para garantir cobertura total
gaia = vizier.query_region(centro, radius=18.0 * u.arcmin, catalog='I/355/gaiadr3')[0]

# Filtra apenas estrelas que possuem todas as medições necessárias para a cor
mask_valid = (~np.isnan(gaia['Gmag'])) & (~np.isnan(gaia['BPmag'])) & (~np.isnan(gaia['RPmag']))
gaia = gaia[mask_valid]

# Aplica a transformação matemática criando a nova coluna 'B_calc'
gaia['B_calc'] = transformar_gaia_para_B(
    gaia['Gmag'].value, gaia['BPmag'].value, gaia['RPmag'].value
)

coords_gaia = SkyCoord(ra=gaia['RA_ICRS'].value * u.deg, dec=gaia['DE_ICRS'].value * u.deg)

# ====================================================================
# 4. Cross-Match (Pareamento)
# ====================================================================
idx_filt, d2d_filt, _ = coords_filt.match_to_catalog_sky(coords_gaia)
mask_filt = d2d_filt < (RAIO_MATCH_ARCSEC * u.arcsec)

x_fit = df_filt['Mag_Inst'].values[mask_filt]
err_fit = df_filt[coluna_erro].values[mask_filt]
y_fit = gaia['B_calc'][idx_filt[mask_filt]].value

idx_tot, d2d_tot, _ = coords_total.match_to_catalog_sky(coords_gaia)
mask_tot = d2d_tot < (RAIO_MATCH_ARCSEC * u.arcsec)

x_tot = df_total['Mag_Inst'].values[mask_tot]
err_tot = df_total[coluna_erro].values[mask_tot]
y_tot = gaia['B_calc'][idx_tot[mask_tot]].value

# ====================================================================
# 5. Ajuste Linear e Cálculo de Limites
# ====================================================================
popt_init, _ = curve_fit(ajuste_linear, x_fit, y_fit)
residuos = y_fit - ajuste_linear(x_fit, *popt_init)

# Rejeita pontos muito fora da reta usando Sigma Clip
residuos_limpos = sigma_clip(residuos, sigma=2.5)
inliers = ~residuos_limpos.mask
outliers = residuos_limpos.mask

# Refaz o ajuste linear apenas com os dados bons
popt, pcov = curve_fit(ajuste_linear, x_fit[inliers], y_fit[inliers])
alfa, c = popt
erro_alfa, erro_c = np.sqrt(np.diag(pcov))

# Salva o arquivo calibrado
df_total['Mag_B_Calibrada'] = ajuste_linear(df_total['Mag_Inst'], alfa, c)
df_total.to_csv(OUTPUT_CSV, index=False)

# Limite Instrumental (5-sigma)
ruido_fundo_total = std_fundo * np.sqrt(area_ap)
fluxo_limite = 5.0 * ruido_fundo_total
mag_inst_limite = -2.5 * np.log10(fluxo_limite / exptime) + ZP_OFFSET
mag_aparente_limite = ajuste_linear(mag_inst_limite, alfa, c)

print("\n--- Resultados do Ajuste (Filtro B transformado do Gaia DR3) ---")
print(f"Estrelas totais pareadas: {len(x_tot)}")
print(f"Estrelas usadas no fit: {np.sum(inliers)}")
print(f"Equação: B_transf = ({alfa:.4f}) * Mag_Inst + ({c:.4f})")
print(f"Limite 5-sigma Calculado: {mag_aparente_limite:.2f} mag\n")

# ====================================================================
# 6. Gráfico Final
# ====================================================================
plt.figure(figsize=(10, 7))

x_min_plot = min(min(x_tot), mag_inst_limite) - 0.5
x_max_plot = max(x_tot) + 0.5
eixo_x_reta = np.linspace(x_min_plot, x_max_plot, 100)
eixo_y_reta = ajuste_linear(eixo_x_reta, alfa, c)

# Plota todas as estrelas
plt.errorbar(x_tot, y_tot, xerr=err_tot, fmt='.', color='gray', alpha=0.3, label='Todas as Estrelas')

# Plota os outliers em vermelho
if np.sum(outliers) > 0:
    plt.errorbar(x_fit[outliers], y_fit[outliers], xerr=err_fit[outliers], fmt='x', color='red', alpha=0.7, label='Outliers (Sigma Clip)')

# Plota as usadas na calibração em azul
plt.errorbar(x_fit[inliers], y_fit[inliers], xerr=err_fit[inliers], fmt='o', color='blue', alpha=0.8, markeredgecolor='k', label='Usadas na Calibração')

# Plota a reta de ajuste
plt.plot(eixo_x_reta, eixo_y_reta, color='black', linewidth=2, label=rf'Fit: $Y = ({alfa:.2f})X + ({c:.2f})$')

# Plota as linhas de limite 5-sigma
plt.axvline(x=mag_inst_limite, color='purple', linestyle='--', alpha=0.6)
plt.axhline(y=mag_aparente_limite, color='purple', linestyle='--', alpha=0.6, label=f'Limite 5$\sigma$: {mag_aparente_limite:.2f} mag')



plt.title('Calibração Filtro B: Instrumental vs Gaia DR3 (Transformado)', fontsize=13)
plt.xlabel('Magnitude Instrumental (Mag_Inst + 25)', fontsize=11)
plt.ylabel('Magnitude Aparente (B de Johnson Calculado via Gaia)', fontsize=11)
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('grafico_calibracao_B_Gaia_Transformado.png', dpi=300)
plt.show()