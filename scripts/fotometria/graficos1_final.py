import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

# ====================================================================
# 1. Parâmetros e Arquivos
# ====================================================================
INPUT_CSV_TOTAL = 'fotometria_final_B.csv'          
INPUT_CSV_FILTRADO = 'fotometria_limpa_B.csv'  
OUTPUT_CSV = 'fotometria_B_calibrada.csv'

RAIO_MATCH_ARCSEC = 3.0  # Aumentado para 3.0 para capturar mais estrelas
ZP_OFFSET = 25.0

def ajuste_linear(x, alfa, c):
    return alfa * x + c

# ====================================================================
# 2. Leitura e Preparação
# ====================================================================
df_total = pd.read_csv(INPUT_CSV_TOTAL)
df_filt = pd.read_csv(INPUT_CSV_FILTRADO)

coluna_erro = 'Erro_Mag' if 'Erro_Mag' in df_total.columns else 'Erro_Mag_Inst'

# Parâmetros da imagem para o limite 5-sigma
std_fundo = df_total['Std_Fundo_Local'].iloc[0] if 'Std_Fundo_Local' in df_total.columns else 10.0
area_ap = df_total['Area_Ap'].iloc[0] if 'Area_Ap' in df_total.columns else 60.0
exptime = df_total['Exptime'].iloc[0] if 'Exptime' in df_total.columns else 1.0

# Aplica o ZP
df_total['Mag_Inst'] = df_total['Mag_Inst'] + ZP_OFFSET
df_filt['Mag_Inst']  = df_filt['Mag_Inst'] + ZP_OFFSET

coords_total = SkyCoord(ra=df_total['RA_deg'].values * u.deg, dec=df_total['Dec_deg'].values * u.deg)
coords_filt  = SkyCoord(ra=df_filt['RA_deg'].values * u.deg,  dec=df_filt['Dec_deg'].values * u.deg)

# ====================================================================
# 3. Busca no Catálogo APASS DR9
# ====================================================================
centro = SkyCoord(ra=df_total['RA_deg'].mean() * u.deg, dec=df_total['Dec_deg'].mean() * u.deg)
vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'Bmag'], row_limit=-1)

apass = vizier.query_region(centro, radius=15.0 * u.arcmin, catalog='II/336/apass9')[0]
# Limite estendido até 18.5 para tentar pegar as estrelas mais fracas do catálogo
apass = apass[(~np.isnan(apass['Bmag'])) & (apass['Bmag'] > 8.0) & (apass['Bmag'] < 18.5)]
coords_apass = SkyCoord(ra=apass['RAJ2000'].value * u.deg, dec=apass['DEJ2000'].value * u.deg)

# ====================================================================
# 4. Pareamento (Cross-Match)
# ====================================================================
idx_filt, d2d_filt, _ = coords_filt.match_to_catalog_sky(coords_apass)
mask_filt = d2d_filt < (RAIO_MATCH_ARCSEC * u.arcsec)

x_fit = df_filt['Mag_Inst'].values[mask_filt]
err_fit = df_filt[coluna_erro].values[mask_filt]
y_fit = apass['Bmag'][idx_filt[mask_filt]].value

idx_tot, d2d_tot, _ = coords_total.match_to_catalog_sky(coords_apass)
mask_tot = d2d_tot < (RAIO_MATCH_ARCSEC * u.arcsec)

x_tot = df_total['Mag_Inst'].values[mask_tot]
err_tot = df_total[coluna_erro].values[mask_tot]
y_tot = apass['Bmag'][idx_tot[mask_tot]].value

# ====================================================================
# 5. Ajuste Linear e Cálculo do Limite
# ====================================================================
popt_init, _ = curve_fit(ajuste_linear, x_fit, y_fit)
residuos = y_fit - ajuste_linear(x_fit, *popt_init)

residuos_limpos = sigma_clip(residuos, sigma=2.5)
inliers = ~residuos_limpos.mask
outliers = residuos_limpos.mask

popt, pcov = curve_fit(ajuste_linear, x_fit[inliers], y_fit[inliers])
alfa, c = popt
erro_alfa, erro_c = np.sqrt(np.diag(pcov))

df_total['Mag_B_Calibrada'] = ajuste_linear(df_total['Mag_Inst'], alfa, c)
df_total.to_csv(OUTPUT_CSV, index=False)

# ---- CÁLCULO DO LIMITE INSTRUMENTAL (5-SIGMA) ----
ruido_fundo_total = std_fundo * np.sqrt(area_ap)
fluxo_limite = 5.0 * ruido_fundo_total
# Transforma o fluxo mínimo detectável em magnitude instrumental (com o ZP embutido)
mag_inst_limite = -2.5 * np.log10(fluxo_limite / exptime) + ZP_OFFSET
# Projeta a magnitude instrumental limite na reta para achar o limite calibrado
mag_aparente_limite = ajuste_linear(mag_inst_limite, alfa, c)

print("\n--- Resultados do Ajuste Linear (APASS B) ---")
print(f"Estrelas totais pareadas no campo: {len(x_tot)}")
print(f"Estrelas filtradas (DS9) usadas no fit: {np.sum(inliers)}")
print(f"Coef. Angular (Alpha): {alfa:.4f} ± {erro_alfa:.4f}")
print(f"Constante (C): {c:.4f} ± {erro_c:.4f}")
print(f"Equação: B_APASS = ({alfa:.4f}) * Mag_Inst + ({c:.4f})")
print(f"Limite Instrumental 5-sigma (Mag_Inst): {mag_inst_limite:.2f}")
print(f"Limite Aparente Calibrado 5-sigma: {mag_aparente_limite:.2f} mag\n")

# ====================================================================
# 6. Gráfico
# ====================================================================
plt.figure(figsize=(10, 7))

# Reta de ajuste expandida até o limite instrumental
x_min_plot = min(min(x_tot), mag_inst_limite) - 0.5
x_max_plot = max(x_tot) + 0.5
eixo_x_reta = np.linspace(x_min_plot, x_max_plot, 100)
eixo_y_reta = ajuste_linear(eixo_x_reta, alfa, c)

# 1. Todas as estrelas
plt.errorbar(x_tot, y_tot, xerr=err_tot, fmt='.', color='gray', 
             alpha=0.3, ecolor='lightgray', elinewidth=0.8, 
             label='Todas as Estrelas (Não usadas no Fit)')

# 2. Rejeitadas (DS9)
if np.sum(outliers) > 0:
    plt.errorbar(x_fit[outliers], y_fit[outliers], xerr=err_fit[outliers], 
                 fmt='x', color='red', alpha=0.7, ecolor='lightcoral', elinewidth=1, 
                 label='Rejeitadas do DS9 (Sigma Clip)')

# 3. Válidas (DS9)
plt.errorbar(x_fit[inliers], y_fit[inliers], xerr=err_fit[inliers], 
             fmt='o', color='blue', alpha=0.8, markeredgecolor='k', 
             ecolor='gray', elinewidth=1.5, capsize=2, 
             label='Usadas na Calibração (Filtro DS9)')

# 4. Reta
plt.plot(eixo_x_reta, eixo_y_reta, color='black', linewidth=2, 
         label=rf'Fit: $Y = ({alfa:.2f})X + ({c:.2f})$')

# 5. Linhas de Limite 5-Sigma
plt.axvline(x=mag_inst_limite, color='purple', linestyle='--', alpha=0.6)
plt.axhline(y=mag_aparente_limite, color='purple', linestyle='--', alpha=0.6, 
            label=f'Limite ($5\sigma$): {mag_aparente_limite:.2f} mag')


plt.title('Calibração Fotométrica B: Instrumental vs APASS DR9', fontsize=13)
plt.xlabel('Magnitude Instrumental (Mag_Inst + 25)', fontsize=11)
plt.ylabel('Magnitude Aparente (APASS B)', fontsize=11)
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('grafico_calibracao_B_APASS_DS9_limite.png', dpi=300)
plt.show()