import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

# ====================================================================
# 1. Configurações
# ====================================================================
ARQUIVO_B_CALIBRADO = 'fotometria_B_calibrada_Gaia.csv'
RAIO_MATCH_ARCSEC = 2.0 

def calcular_temperatura_empirica(cor_B_G):
    """
    Estima a temperatura efetiva (K) baseada no índice de cor.
    Utiliza uma aproximação da equação de Ballesteros.
    """
    termo1 = 1 / (0.92 * cor_B_G + 1.7)
    termo2 = 1 / (0.92 * cor_B_G + 0.62)
    return 4600 * (termo1 + termo2)

# ====================================================================
# 2. Leitura dos Dados e Consulta ao Gaia
# ====================================================================
print("Carregando fotometria B e consultando Paralaxes no Gaia DR3...")
df_B = pd.read_csv(ARQUIVO_B_CALIBRADO)
coords_B = SkyCoord(ra=df_B['RA_deg'].values * u.deg, dec=df_B['Dec_deg'].values * u.deg)

centro = SkyCoord(ra=df_B['RA_deg'].mean() * u.deg, dec=df_B['Dec_deg'].mean() * u.deg)

# Solicitamos a Paralaxe (Plx) em vez de procurar a coluna de luminosidade perdida
vizier = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag', 'Plx'], row_limit=-1)
gaia = vizier.query_region(centro, radius=18.0 * u.arcmin, catalog='I/355/gaiadr3')[0]

# Filtra apenas estrelas com paralaxe válida e positiva (necessário para distância)
gaia = gaia[(~np.isnan(gaia['Plx'])) & (gaia['Plx'] > 0)]
coords_gaia = SkyCoord(ra=gaia['RA_ICRS'].value * u.deg, dec=gaia['DE_ICRS'].value * u.deg)

# ====================================================================
# 3. Cross-Match e Cálculos Astrofísicos
# ====================================================================
idx, d2d, _ = coords_B.match_to_catalog_sky(coords_gaia)
mask = d2d < (RAIO_MATCH_ARCSEC * u.arcsec)

mag_B = df_B['Mag_B_Calibrada'].values[mask]
mag_G = gaia['Gmag'][idx[mask]].value
paralaxe_mas = gaia['Plx'][idx[mask]].value

# A. Índice de Cor
indice_cor = mag_B - mag_G

# B. Temperatura
temperatura = calcular_temperatura_empirica(indice_cor)

# C. Luminosidade a partir da Paralaxe
# 1. Distância em parsecs (Paralaxe do Gaia está em milissegundos de arco)
distancia_pc = 1000.0 / paralaxe_mas
# 2. Magnitude Absoluta (M_G)
mag_abs_G = mag_G - 5 * np.log10(distancia_pc) + 5
# 3. Luminosidade em relação ao Sol (Magnitude absoluta do Sol em G é ~4.66)
luminosidade = 10 ** ((4.66 - mag_abs_G) / 2.5)

print(f"Sucesso! {np.sum(mask)} estrelas pareadas e processadas.")

# ====================================================================
# 4. Gráfico 1: Diagrama de Índice de Cor B vs (B-G)
# ====================================================================
plt.figure(figsize=(8, 6))
plt.scatter(indice_cor, mag_B, c=indice_cor, cmap='coolwarm', s=15, alpha=0.8, edgecolor='k', linewidth=0.2)

plt.gca().invert_yaxis()

plt.title('Diagrama Cor-Magnitude: $B$ vs $(B-G)$')
plt.xlabel(r'Índice de Cor $(B - G)$ [Mais Quente $\leftarrow$ $\rightarrow$ Mais Frio]')
plt.ylabel('Magnitude Aparente $B$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.colorbar(label='Índice $(B-G)$')
plt.tight_layout()
plt.savefig('diagrama_cor_magnitude.png', dpi=300)
plt.show()

# ====================================================================
# 5. Gráfico 2: Temperatura Empírica vs Luminosidade (HR)
# ====================================================================
plt.figure(figsize=(8, 6))

plt.scatter(temperatura, luminosidade, c=temperatura, cmap='Spectral', s=20, alpha=0.8, edgecolor='k', linewidth=0.3)

#plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.yscale('log')

plt.title('Diagrama HR Empírico (via Paralaxe Gaia DR3)')
plt.xlabel('Temperatura Empírica Estimada (K)')
plt.ylabel(r'Luminosidade ($L/L_\odot$)')
plt.grid(True, linestyle=':', alpha=0.6, which='both')
plt.tight_layout()
#plt.xlim(-)
plt.savefig('diagrama_HR_temperatura_luminosidade.png', dpi=300)
plt.show()