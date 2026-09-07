import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

# Configurações Iniciais (AJUSTE OS NOMES DOS SEUS ARQUIVOS AQUI)
ARQUIVO_B = 'resultado_fotometria_B_calibrada.csv' # Arquivo do filtro B
ARQUIVO_G = 'resultado_fotometria_G_calibrada.csv' # Arquivo do filtro G
RA_ALVO, DEC_ALVO = 276.5627, -14.8479
RAIO_VIZIER = 15 * u.arcmin
TOLERANCIA_MATCH = 2.5 * u.arcsec

# 1. Carregar os Dados e Definir Coordenadas
df_b = pd.read_csv(ARQUIVO_B)
df_g = pd.read_csv(ARQUIVO_G)
coord_alvo = SkyCoord(ra=RA_ALVO*u.deg, dec=DEC_ALVO*u.deg)

coords_b = SkyCoord(ra=df_b['RA_deg'].values*u.deg, dec=df_b['Dec_deg'].values*u.deg)
coords_g = SkyCoord(ra=df_g['RA_deg'].values*u.deg, dec=df_g['Dec_deg'].values*u.deg)

# 2. Cross-Match Local (Filtro B vs G)
idx_g, d2d_bg, _ = coords_b.match_to_catalog_sky(coords_g)
valid_bg = d2d_bg < TOLERANCIA_MATCH

df_obs = df_b[valid_bg].copy().reset_index(drop=True)
df_obs.rename(columns={'Mag_Inst': 'Mag_Inst_B'}, inplace=True)
df_obs['Mag_Inst_G'] = df_g['Mag_Inst'].iloc[idx_g[valid_bg]].values
coords_obs = coords_b[valid_bg]

print(f"Estrelas combinadas (B + G): {len(df_obs)}")

# 3. Consulta ao Gaia DR3
print("Consultando o Gaia DR3 no VizieR...")
v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'BPmag', 'Gmag'], row_limit=5000)
tabela_gaia = v.query_region(coord_alvo, radius=RAIO_VIZIER, catalog='I/355/gaiadr3')[0].to_pandas()
coords_gaia = SkyCoord(ra=tabela_gaia['RA_ICRS'].values*u.deg, dec=tabela_gaia['DE_ICRS'].values*u.deg)

# 4. Cross-Match com o Gaia para Calibração de Ponto Zero
idx_gaia, d2d_gaia, _ = coords_obs.match_to_catalog_sky(coords_gaia)
valid_gaia = d2d_gaia < TOLERANCIA_MATCH

# Calcula os offsets 
offset_B = tabela_gaia['BPmag'].iloc[idx_gaia[valid_gaia]].values - df_obs['Mag_Inst_B'][valid_gaia].values
offset_G = tabela_gaia['Gmag'].iloc[idx_gaia[valid_gaia]].values - df_obs['Mag_Inst_G'][valid_gaia].values

zp_B = np.nanmedian(offset_B)
zp_G = np.nanmedian(offset_G)
print(f"Zero Point B (BP): {zp_B:.3f} | Zero Point G (G): {zp_G:.3f}")

# 5. Magnitudes Calibradas e Índice de Cor (B - G)
df_obs['Mag_Ap_B'] = df_obs['Mag_Inst_B'] + zp_B
df_obs['Mag_Ap_G'] = df_obs['Mag_Inst_G'] + zp_G
df_obs['Cor_B_G'] = df_obs['Mag_Ap_B'] - df_obs['Mag_Ap_G']
df_obs.to_csv('resultado_fotometria_calibrada_BG.csv', index=False)

# 6. Identificar o Alvo (LS 5039)
idx_alvo = np.argmin(coords_obs.separation(coord_alvo))

# 7. Plotagem do Diagrama HR (B - G)
plt.figure(figsize=(9, 7))

# Plot das estrelas do campo
plt.scatter(df_obs['Cor_B_G'], df_obs['Mag_Ap_G'], c='black', alpha=0.6, s=25, label='Estrelas do Campo')

# Destaque do Alvo (LS 5039)
plt.scatter(df_obs['Cor_B_G'].iloc[idx_alvo], df_obs['Mag_Ap_G'].iloc[idx_alvo], 
            c='red', s=200, marker='*', edgecolor='black', label='LS 5039')

# Linha de Tendência da Sequência Principal
# Ajuste flexível dos limites baseado nos dados reais para evitar distorções
lim_inf = df_obs['Cor_B_G'].quantile(0.10)
lim_sup = df_obs['Cor_B_G'].quantile(0.90)
mascara_tendencia = (df_obs['Cor_B_G'] > lim_inf) & (df_obs['Cor_B_G'] < lim_sup)

if mascara_tendencia.sum() > 5: # Garante que há pontos suficientes para traçar a linha
    cor_limpa = df_obs['Cor_B_G'][mascara_tendencia]
    mag_limpa = df_obs['Mag_Ap_G'][mascara_tendencia]
    
    coeficientes = np.polyfit(cor_limpa, mag_limpa, 1) # Ajuste linear costuma ser mais estável em campos reduzidos
    polinomio = np.poly1d(coeficientes)
    
    x_linha = np.linspace(lim_inf, lim_sup, 100)
    y_linha = polinomio(x_linha)
    plt.plot(x_linha, y_linha, color='dodgerblue', linestyle='--', linewidth=2.5, label='Tendência (Sequência Principal)')

plt.gca().invert_yaxis() # Magnitudes mais brilhantes para cima
plt.xlabel('Índice de Cor $(B - G)$', fontsize=13)
plt.ylabel('Magnitude Aparente $G$', fontsize=13)
plt.title('Diagrama HR — Campo LS 5039', fontsize=14, pad=15)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('diagrama_HR_BG_com_linha.png', dpi=300)
plt.show()