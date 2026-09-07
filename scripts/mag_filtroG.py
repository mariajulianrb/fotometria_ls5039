import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

# 1. Carrega os Dados Brutos da Fotometria
df_bruto = pd.read_csv('fotometria_bruta_G.csv')

std_fundo = df_bruto['Std_Fundo'].iloc[0]
area_ap = df_bruto['Area_Ap'].iloc[0]
exptime = df_bruto['Exptime'].iloc[0]

# Coordenadas das estrelas na imagem e o centro do campo
coords_imagem = SkyCoord(ra=df_bruto['RA_deg'].values * u.deg, dec=df_bruto['Dec_deg'].values * u.deg)
centro_coord = SkyCoord(ra=df_bruto['RA_deg'].mean() * u.deg, dec=df_bruto['Dec_deg'].mean() * u.deg)

# 2. Consulta ao catálogo Gaia DR3 via VizieR (Banda G)
print("Consultando VizieR (Gaia DR3, Banda G)...")
vizier = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag'], row_limit=-1)
catalogo = vizier.query_region(centro_coord, radius=15 * u.arcmin, catalog='I/355/gaiadr3')[0]

# Limpeza: remove NaNs e filtra pelo brilho (10 < Gmag < 18)
catalogo = catalogo[~np.isnan(catalogo['Gmag'])]
catalogo = catalogo[(catalogo['Gmag'] > 10.0) & (catalogo['Gmag'] < 18.0)]

coords_catalogo = SkyCoord(ra=catalogo['RA_ICRS'] * u.deg, dec=catalogo['DE_ICRS'] * u.deg)

# 3. Cross-Matching e Zero Point
idx_catalogo, d2d, _ = coords_imagem.match_to_catalog_sky(coords_catalogo)

# Mantém apenas cruzamentos com menos de 2 arcossegundos
pares = d2d < (2.0 * u.arcsec)
mag_inst = df_bruto['Mag_Inst'].values[pares]
mag_cat = catalogo['Gmag'][idx_catalogo[pares]]

# Calcula o Zero Point eliminando discrepâncias (sigma clipping)
diferencas = mag_cat - mag_inst
diferencas_limpas = sigma_clip(diferencas, sigma=2.5)

zero_point = np.ma.median(diferencas_limpas)
desvio_zp = np.ma.std(diferencas_limpas)

print(f"Estrelas pareadas: {len(mag_inst)}")
print(f"Zero Point (ZP): {zero_point:.4f} ± {desvio_zp:.4f} mag")

# 4. Calibração e Magnitude Limite
ruido = std_fundo * np.sqrt(area_ap)
mag_inst_limite = -2.5 * np.log10((5.0 * ruido) / exptime)
mag_limite = mag_inst_limite + zero_point

# Salva os resultados
df_bruto['Mag_G_Calibrada'] = df_bruto['Mag_Inst'] + zero_point
df_bruto.to_csv('resultado_fotometria_G_calibrada.csv', index=False)

# 5. Gráficos Diagnósticos Simplificados
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

inliers = ~diferencas_limpas.mask
outliers = diferencas_limpas.mask

# P1: Calibração Linear
axes[0].scatter(mag_inst[inliers], mag_cat[inliers], c='royalblue', label='Inliers')
if outliers.any():
    axes[0].scatter(mag_inst[outliers], mag_cat[outliers], c='red', marker='x', label='Outliers')

x_line = np.array([mag_inst.min(), mag_inst.max()])
axes[0].plot(x_line, x_line + zero_point, color='darkorange', ls='--', lw=2, label=f'ZP = {zero_point:.2f}')
axes[0].set(xlabel='Mag Instrumental', ylabel='Mag Gaia (G)', title='Calibração ZP')
axes[0].invert_xaxis()
axes[0].invert_yaxis()
axes[0].legend()
axes[0].grid(ls='--', alpha=0.5)

# P2: Resíduos
residuos = mag_cat[inliers] - (mag_inst[inliers] + zero_point)
axes[1].scatter(mag_cat[inliers], residuos, c='purple', alpha=0.6)
axes[1].axhline(0, c='black', ls='--')
axes[1].set(xlabel='Mag Gaia (G)', ylabel='Resíduo', title='Resíduos do ZP')
axes[1].invert_xaxis()
axes[1].grid(ls='--', alpha=0.5)

# P3: Histograma
axes[2].hist(df_bruto['Mag_G_Calibrada'], bins=30, color='steelblue', edgecolor='black')
axes[2].axvline(mag_limite, c='red', ls='--', lw=2, label=f'Lim 5-sigma ({mag_limite:.2f})')
axes[2].set(xlabel='Mag Calibrada (G)', ylabel='Nº de Estrelas', title='Distribuição')
axes[2].legend()
axes[2].grid(ls='--', alpha=0.5)

plt.tight_layout()
plt.show()