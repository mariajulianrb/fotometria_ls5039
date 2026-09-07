import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

# 1. Carrega os Dados Brutos da Fotometria
df_bruto = pd.read_csv('fotometria_bruta_B.csv')

std_fundo = df_bruto['Std_Fundo'].iloc[0]
area_ap = df_bruto['Area_Ap'].iloc[0]
exptime = df_bruto['Exptime'].iloc[0]

coords_imagem = SkyCoord(ra=df_bruto['RA_deg'].values * u.deg, dec=df_bruto['Dec_deg'].values * u.deg)
centro_coord = SkyCoord(ra=df_bruto['RA_deg'].mean() * u.deg, dec=df_bruto['Dec_deg'].mean() * u.deg)

# 2. Consulta ao catálogo APASS DR9 via VizieR (Banda B)
print("Consultando VizieR (APASS DR9, Banda B)...")
vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'Bmag'], row_limit=-1)
catalogo = vizier.query_region(centro_coord, radius=15 * u.arcmin, catalog='II/336/apass9')[0]

# Limpeza: remove NaNs e filtra pelo brilho (10 < Bmag < 17)
catalogo = catalogo[~np.isnan(catalogo['Bmag'])]
catalogo = catalogo[(catalogo['Bmag'] > 10.0) & (catalogo['Bmag'] < 17.0)]

# CORREÇÃO APLICADA AQUI (Uso do np.array para evitar conflito de unidades "deg2")
coords_catalogo = SkyCoord(ra=np.array(catalogo['RAJ2000']) * u.deg, dec=np.array(catalogo['DEJ2000']) * u.deg)

# 3. Cross-Matching e Zero Point
idx_catalogo, d2d, _ = coords_imagem.match_to_catalog_sky(coords_catalogo)

pares = d2d < (2.0 * u.arcsec)
mag_inst = df_bruto['Mag_Inst'].values[pares]
mag_cat = np.array(catalogo['Bmag'][idx_catalogo[pares]])

diferencas = mag_cat - mag_inst
diferencas_limpas = sigma_clip(diferencas, sigma=2.5)

# Cálculo do Zero Point, Desvio Padrão e Erro Padrão
zero_point = np.ma.median(diferencas_limpas)
desvio_zp = np.ma.std(diferencas_limpas)
n_inliers = diferencas_limpas.count() # Conta apenas os pontos não descartados pelo sigma_clip

erro_zp = desvio_zp / np.sqrt(n_inliers)

print(f"Estrelas pareadas: {len(mag_inst)} (Inliers utilizados: {n_inliers})")
print(f"Zero Point (ZP): {zero_point:.4f} ± {erro_zp:.4f} mag (Desvio: {desvio_zp:.4f})")

# 4. Calibração e Magnitude Limite
ruido = std_fundo * np.sqrt(area_ap)
mag_inst_limite = -2.5 * np.log10((5.0 * ruido) / exptime)
mag_limite = mag_inst_limite + zero_point

df_bruto['Mag_B_Calibrada'] = df_bruto['Mag_Inst'] + zero_point
df_bruto.to_csv('resultado_fotometria_B_calibrada.csv', index=False)

# 5. Gráficos Diagnósticos Simplificados
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Tratamento seguro da máscara do sigma_clip
if np.ma.is_masked(diferencas_limpas):
    outliers = diferencas_limpas.mask
    inliers = ~outliers
else:
    outliers = np.zeros_like(diferencas_limpas, dtype=bool)
    inliers = ~outliers

# P1: Calibração Linear
axes[0].scatter(mag_inst[inliers], mag_cat[inliers], c='royalblue', label='Inliers')
if outliers.any():
    axes[0].scatter(mag_inst[outliers], mag_cat[outliers], c='red', marker='x', label='Outliers')

x_line = np.array([mag_inst.min(), mag_inst.max()])
axes[0].plot(x_line, x_line + zero_point, color='darkorange', ls='--', lw=2, label=f'ZP = {zero_point:.2f}')
axes[0].set(xlabel='Mag Instrumental', ylabel='Mag APASS (B)', title='Calibração ZP (Banda B)')
axes[0].invert_xaxis()
axes[0].invert_yaxis()
axes[0].legend()
axes[0].grid(ls='--', alpha=0.5)

# P2: Resíduos
residuos = mag_cat[inliers] - (mag_inst[inliers] + zero_point)
axes[1].scatter(mag_cat[inliers], residuos, c='purple', alpha=0.6)
axes[1].axhline(0, c='black', ls='--')
axes[1].set(xlabel='Mag APASS (B)', ylabel='Resíduo', title='Resíduos do ZP (Banda B)')
axes[1].invert_xaxis()
axes[1].grid(ls='--', alpha=0.5)

# P3: Histograma
axes[2].hist(df_bruto['Mag_B_Calibrada'], bins=30, color='steelblue', edgecolor='black')
axes[2].axvline(mag_limite, c='red', ls='--', lw=2, label=f'Lim 5-sigma ({mag_limite:.2f})')
axes[2].set(xlabel='Mag Calibrada (B)', ylabel='Nº de Estrelas', title='Distribuição (Banda B)')
axes[2].legend()
axes[2].grid(ls='--', alpha=0.5)

plt.tight_layout()
plt.show()