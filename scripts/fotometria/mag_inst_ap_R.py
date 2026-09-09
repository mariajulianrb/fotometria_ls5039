import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

# ====================================================================
# 1. Arquivos e Configurações (Filtro R via Gaia DR3)
# ====================================================================
INPUT_CSV_TOTAL = 'fotometria_final_R.csv'
OUTPUT_CSV = 'fotometria_R_calibrada_Gaia.csv'

RAIO_MATCH_ARCSEC = 4.0 
ZP_OFFSET = 25.0
NUM_ESTRELAS_FIT = 500  # Usar apenas as N mais brilhantes para a reta

def ajuste_linear(x, alfa, c):
    return alfa * x + c

def transformar_gaia_para_R(G, BP, RP):
    """Transformação empírica de Gaia para a banda R (Cousins)."""
    bp_rp = BP - RP
    R_minus_G = -0.0107 - 0.1491 * bp_rp - 0.0084 * (bp_rp**2)
    return G + R_minus_G

# ====================================================================
# 2. Leitura e Preparação dos Dados Instrumentais
# ====================================================================
df_total = pd.read_csv(INPUT_CSV_TOTAL)
coluna_erro = 'Erro_Mag' if 'Erro_Mag' in df_total.columns else 'Erro_Mag_Inst'

std_fundo = df_total['Std_Fundo_Local'].iloc[0] if 'Std_Fundo_Local' in df_total.columns else 10.0
area_ap = df_total['Area_Ap'].iloc[0] if 'Area_Ap' in df_total.columns else 60.0
exptime = df_total['Exptime'].iloc[0] if 'Exptime' in df_total.columns else 1.0

# Aplica o Zero Point
df_total['Mag_Inst'] = df_total['Mag_Inst'] + ZP_OFFSET
coords_total = SkyCoord(ra=df_total['RA_deg'].values * u.deg, dec=df_total['Dec_deg'].values * u.deg)

# ====================================================================
# 3. Consulta ao Gaia DR3
# ====================================================================
centro = SkyCoord(ra=df_total['RA_deg'].mean() * u.deg, dec=df_total['Dec_deg'].mean() * u.deg)
vizier = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag', 'BPmag', 'RPmag'], row_limit=-1)
gaia = vizier.query_region(centro, radius=18.0 * u.arcmin, catalog='I/355/gaiadr3')[0]

# Filtra apenas estrelas que possuem G, BP e RP (Isso causa o limite visível ~16 mag do catálogo)
mask_valid = (~np.isnan(gaia['Gmag'])) & (~np.isnan(gaia['BPmag'])) & (~np.isnan(gaia['RPmag']))
gaia = gaia[mask_valid]

gaia['R_calc'] = transformar_gaia_para_R(gaia['Gmag'].value, gaia['BPmag'].value, gaia['RPmag'].value)
coords_gaia = SkyCoord(ra=gaia['RA_ICRS'].value * u.deg, dec=gaia['DE_ICRS'].value * u.deg)

# ====================================================================
# 4. Cross-Match e Seleção das 500 mais Brilhantes
# ====================================================================
idx_tot, d2d_tot, _ = coords_total.match_to_catalog_sky(coords_gaia)
mask_tot = d2d_tot < (RAIO_MATCH_ARCSEC * u.arcsec)

# Cria um DataFrame temporário só com as pareadas para podermos ordenar
df_match = pd.DataFrame({
    'Mag_Inst': df_total['Mag_Inst'].values[mask_tot],
    'Erro_Inst': df_total[coluna_erro].values[mask_tot],
    'Mag_Cat': gaia['R_calc'][idx_tot[mask_tot]].value
})

# Ordena do mais brilhante (menor mag) para o mais fraco
df_match = df_match.sort_values(by='Mag_Inst').reset_index(drop=True)

x_tot = df_match['Mag_Inst'].values
y_tot = df_match['Mag_Cat'].values
err_tot = df_match['Erro_Inst'].values

# Isola estritamente as 500 mais brilhantes para o ajuste
df_fit = df_match.head(NUM_ESTRELAS_FIT)
x_fit = df_fit['Mag_Inst'].values
y_fit = df_fit['Mag_Cat'].values
err_fit = df_fit['Erro_Inst'].values

# ====================================================================
# 5. Ajuste Linear e Limite Instrumental
# ====================================================================
popt_init, _ = curve_fit(ajuste_linear, x_fit, y_fit)
residuos = y_fit - ajuste_linear(x_fit, *popt_init)

# Limpa outliers apenas das 500 estrelas selecionadas
residuos_limpos = sigma_clip(residuos, sigma=2.5)
inliers = ~residuos_limpos.mask
outliers = residuos_limpos.mask

# Ajuste final
popt, pcov = curve_fit(ajuste_linear, x_fit[inliers], y_fit[inliers])
alfa, c = popt

# Calibra o catálogo COMPLETO original
df_total['Mag_R_Calibrada'] = ajuste_linear(df_total['Mag_Inst'], alfa, c)
df_total.to_csv(OUTPUT_CSV, index=False)

# Cálculo independente do limite 5-sigma (usando o ruído de fundo da imagem)
ruido_fundo_total = std_fundo * np.sqrt(area_ap)
fluxo_limite = 5.0 * ruido_fundo_total
mag_inst_limite = -2.5 * np.log10(fluxo_limite / exptime) + ZP_OFFSET
mag_aparente_limite = ajuste_linear(mag_inst_limite, alfa, c)

print("\n--- Resultados do Ajuste (Filtro R via Gaia DR3) ---")
print(f"Total pareado (Gaia com BP/RP): {len(x_tot)}")
print(f"Estrelas fornecidas pro fit: {len(x_fit)} mais brilhantes")
print(f"Estrelas efetivamente usadas (inliers): {np.sum(inliers)}")
print(f"Equação: R_transf = ({alfa:.4f}) * Mag_Inst + ({c:.4f})")
print(f"Limite 5-sigma Calculado: {mag_aparente_limite:.2f} mag\n")

# ====================================================================
# 6. Gráfico Final
# ====================================================================
plt.figure(figsize=(10, 7))

x_min_plot = min(min(x_tot), mag_inst_limite) - 0.5
x_max_plot = max(x_tot) + 0.5
eixo_x_reta = np.linspace(x_min_plot, x_max_plot, 100)
eixo_y_reta = ajuste_linear(eixo_x_reta, alfa, c)

# Plota TODAS as pareadas em cinza (como pano de fundo)
plt.errorbar(x_tot, y_tot, xerr=err_tot, fmt='.', color='gray', alpha=0.3, label='Todas as Pareadas no Gaia')

# Plota os outliers em vermelho (dentre as 500 mais brilhantes)
if np.sum(outliers) > 0:
    plt.errorbar(x_fit[outliers], y_fit[outliers], xerr=err_fit[outliers], fmt='x', color='red', alpha=0.7, label='Outliers Rejeitados')

# Plota as usadas no fit final em azul
plt.errorbar(x_fit[inliers], y_fit[inliers], xerr=err_fit[inliers], fmt='o', color='blue', alpha=0.7, markeredgecolor='k', label=f'Top {NUM_ESTRELAS_FIT} Usadas no Fit')

plt.plot(eixo_x_reta, eixo_y_reta, color='black', linewidth=2, label=rf'Fit: $Y = ({alfa:.2f})X + ({c:.2f})$')

plt.axvline(x=mag_inst_limite, color='purple', linestyle='--', alpha=0.6)
plt.axhline(y=mag_aparente_limite, color='purple', linestyle='--', alpha=0.6, label=f'Limite 5$\sigma$: {mag_aparente_limite:.2f} mag')

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.title('Calibração Filtro R: Top 500 Estrelas vs Limite Teórico', fontsize=13)
plt.xlabel('Magnitude Instrumental (Mag_Inst + 25)', fontsize=11)
plt.ylabel('Magnitude Aparente (R de Cousins Calculado via Gaia)', fontsize=11)
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('grafico_calibracao_R_Gaia_Top500.png', dpi=300)
plt.show()