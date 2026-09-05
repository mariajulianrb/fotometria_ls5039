import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import os

# 1. Carrega os Dados Brutos da Fotometria
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
caminho_csv = os.path.join(diretorio_atual, 'fotometria_bruta_B.csv')

df_bruto = pd.read_csv(caminho_csv)

std_fundo = df_bruto['Std_Fundo'].iloc[0]
area_ap = df_bruto['Area_Ap'].iloc[0]
exptime = df_bruto['Exptime'].iloc[0]

# Pandas Series usam .values (plural)
coords_imagem = SkyCoord(ra=df_bruto['RA_deg'].values * u.deg, dec=df_bruto['Dec_deg'].values * u.deg)

# 2. Consulta ao Catálogo UCAC4 via VizieR
centro_coord = SkyCoord(ra=df_bruto['RA_deg'].mean() * u.deg, dec=df_bruto['Dec_deg'].mean() * u.deg)

print("Consultando o catálogo UCAC4 no VizieR...")
vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'Bmag'], row_limit=-1)
catalogo = vizier.query_region(centro_coord, radius=15 * u.arcmin, catalog='I/322A/out')[0]
catalogo = catalogo[~np.isnan(catalogo['Bmag'])]

# Astropy MaskedColumns usam .value (singular)
coords_catalogo = SkyCoord(ra=catalogo['RAJ2000'].value * u.deg, dec=catalogo['DEJ2000'].value * u.deg)

# 3. Cross-Matching e Zero Point
idx_catalogo, d2d, _ = coords_imagem.match_to_catalog_sky(coords_catalogo)
pares_validos = d2d < (2.0 * u.arcsec)

mag_inst_pareada = df_bruto['Mag_Inst'].values[pares_validos]
mag_aparente_pareada = catalogo['Bmag'][idx_catalogo[pares_validos]]

diferencas = mag_aparente_pareada - mag_inst_pareada
diferencas_limpas = sigma_clip(diferencas, sigma=2.5)

zero_point = np.ma.median(diferencas_limpas)
desvio_padrao_zp = np.ma.std(diferencas_limpas)

print(f"Estrelas pareadas: {len(mag_inst_pareada)}")
print(f"Zero Point (ZP): {zero_point:.4f} ± {desvio_padrao_zp:.4f} mag")

# 4. Calibração e Magnitude Limite 5-sigma
ruido_abertura = std_fundo * np.sqrt(area_ap)
flux_5sigma = 5.0 * ruido_abertura
mag_inst_5sigma = -2.5 * np.log10(flux_5sigma / exptime)
mag_limite = mag_inst_5sigma + zero_point

df_bruto['Mag_B_Calibrada'] = df_bruto['Mag_Inst'] + zero_point
df_bruto.to_csv('resultado_fotometria_B_completo.csv', index=False)

# 5. Gráficos Diagnósticos
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

mascara_inliers = ~diferencas_limpas.mask
mascara_outliers = diferencas_limpas.mask

axes[0].scatter(mag_inst_pareada[mascara_inliers], mag_aparente_pareada[mascara_inliers],
                color='royalblue', alpha=0.7, edgecolor='k', label='Inliers Válidos')
if mascara_outliers.any():
    axes[0].scatter(mag_inst_pareada[mascara_outliers], mag_aparente_pareada[mascara_outliers],
                    color='red', alpha=0.7, marker='x', label='Outliers Descartados')

x_line = np.linspace(min(mag_inst_pareada), max(mag_inst_pareada), 100)
axes[0].plot(x_line, x_line + zero_point, color='darkorange', linestyle='--', linewidth=2,
             label=f'Ajuste ZP = {zero_point:.2f} $\\pm$ {desvio_padrao_zp:.2f}')
axes[0].set_xlabel('Magnitude Instrumental')
axes[0].set_ylabel('Magnitude Aparente ($B_{mag}$ UCAC4)')
axes[0].set_title('Calibração de Zero Point')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].invert_xaxis()
axes[0].invert_yaxis()

axes[1].hist(df_bruto['Mag_B_Calibrada'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].axvline(x=mag_limite, color='red', linestyle='--', linewidth=2,
                label=f'Limite 5-sigma ({mag_limite:.2f} mag)')
axes[1].set_xlabel('Magnitude Calibrada ($B$)')
axes[1].set_ylabel('Número de Estrelas')
axes[1].set_title('Distribuição de Magnitudes e Limite de Detecção')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()