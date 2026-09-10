import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier
from astropy.stats import sigma_clip

# ======================================================================
# Configurações Iniciais
# ======================================================================
diretorio_atual = os.getcwd()
ARQUIVO_B = os.path.join(diretorio_atual, 'resultado_fotometria_B_calibrada.csv') 
ARQUIVO_G = os.path.join(diretorio_atual, 'resultado_fotometria_G_calibrada.csv') 

RA_ALVO, DEC_ALVO = 276.5627, -14.8479
RAIO_VIZIER = 15 * u.arcmin
TOLERANCIA_MATCH = 2.5 * u.arcsec
ZP_OFFSET = 25.0

# ======================================================================
# 1. Carregar os Dados e Definir Coordenadas
# ======================================================================
df_b = pd.read_csv(ARQUIVO_B)
df_g = pd.read_csv(ARQUIVO_G)

df_b.columns = df_b.columns.str.strip()
df_g.columns = df_g.columns.str.strip()

# Aplicar o Zero Point base (ZP = 25)
df_b['Mag_Inst'] = df_b['Mag_Inst'] + ZP_OFFSET
df_g['Mag_Inst'] = df_g['Mag_Inst'] + ZP_OFFSET

coord_alvo = SkyCoord(ra=RA_ALVO * u.deg, dec=DEC_ALVO * u.deg)

coords_b = SkyCoord(ra=df_b['RA_deg'].values * u.deg, dec=df_b['Dec_deg'].values * u.deg)
coords_g = SkyCoord(ra=df_g['RA_deg'].values * u.deg, dec=df_g['Dec_deg'].values * u.deg)

# ======================================================================
# 2. Cross-Match entre as Imagens (Filtro B vs G)
# ======================================================================
idx_g, d2d_bg, _ = coords_b.match_to_catalog_sky(coords_g)
valid_bg = d2d_bg < TOLERANCIA_MATCH

df_obs = df_b[valid_bg].copy().reset_index(drop=True)
df_obs.rename(columns={'Mag_Inst': 'Mag_Inst_B', 'Erro_Mag': 'Erro_Mag_B'}, inplace=True)

df_obs['Mag_Inst_G'] = df_g['Mag_Inst'].iloc[idx_g[valid_bg]].values
col_erro_g = 'Erro_Mag' if 'Erro_Mag' in df_g.columns else 'Erro_Mag_Inst'
df_obs['Erro_Mag_G'] = df_g[col_erro_g].iloc[idx_g[valid_bg]].values

coords_obs = coords_b[valid_bg]
print(f"Estrelas combinadas (B + G): {len(df_obs)}")

# ======================================================================
# 3. Consulta ao APASS DR9 e Gaia DR3 no VizieR
# ======================================================================
print("Consultando APASS DR9 e Gaia DR3 no VizieR...")

v_apass = Vizier(columns=['RAJ2000', 'DEJ2000', 'Bmag'], row_limit=-1)
tabela_apass = v_apass.query_region(coord_alvo, radius=RAIO_VIZIER, catalog='II/336/apass9')[0].to_pandas()
tabela_apass = tabela_apass.dropna(subset=['Bmag']) 
coords_apass = SkyCoord(ra=tabela_apass['RAJ2000'].values * u.deg, dec=tabela_apass['DEJ2000'].values * u.deg)

# Incluímos o erro da paralaxe (e_Plx) novamente
v_gaia = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag', 'Plx', 'e_Plx'], row_limit=-1)
tabela_gaia = v_gaia.query_region(coord_alvo, radius=RAIO_VIZIER, catalog='I/355/gaiadr3')[0].to_pandas()
tabela_gaia = tabela_gaia.dropna(subset=['Gmag', 'Plx', 'e_Plx'])
coords_gaia = SkyCoord(ra=tabela_gaia['RA_ICRS'].values * u.deg, dec=tabela_gaia['DE_ICRS'].values * u.deg)

# ======================================================================
# 4. Cross-Match e Calibração dos Pontos Zeros
# ======================================================================
idx_apass, d2d_apass, _ = coords_obs.match_to_catalog_sky(coords_apass)
valid_apass = d2d_apass < TOLERANCIA_MATCH
offset_B_bruto = tabela_apass['Bmag'].iloc[idx_apass[valid_apass]].values - df_obs['Mag_Inst_B'][valid_apass].values
offset_B_limpo = sigma_clip(offset_B_bruto, sigma=2.5)
zp_B_residual = np.ma.median(offset_B_limpo)

idx_gaia, d2d_gaia, _ = coords_obs.match_to_catalog_sky(coords_gaia)
valid_gaia = d2d_gaia < TOLERANCIA_MATCH
offset_G_bruto = tabela_gaia['Gmag'].iloc[idx_gaia[valid_gaia]].values - df_obs['Mag_Inst_G'][valid_gaia].values
offset_G_limpo = sigma_clip(offset_G_bruto, sigma=2.5)
zp_G_residual = np.ma.median(offset_G_limpo)

print(f"Residual Zero Point B (APASS B): {zp_B_residual:.3f} | Residual Zero Point G (Gaia G): {zp_G_residual:.3f}")

# Capturar Paralaxe e seu Erro
df_obs['Paralaxe_Gaia'] = np.nan
df_obs['Erro_Paralaxe_Gaia'] = np.nan
df_obs.loc[valid_gaia, 'Paralaxe_Gaia'] = tabela_gaia['Plx'].iloc[idx_gaia[valid_gaia]].values
df_obs.loc[valid_gaia, 'Erro_Paralaxe_Gaia'] = tabela_gaia['e_Plx'].iloc[idx_gaia[valid_gaia]].values

# ======================================================================
# 5. Magnitudes Calibradas, Cor e Distância
# ======================================================================
df_obs['Mag_Ap_B'] = df_obs['Mag_Inst_B'] + zp_B_residual
df_obs['Mag_Ap_G'] = df_obs['Mag_Inst_G'] + zp_G_residual
df_obs['Cor_B_G'] = df_obs['Mag_Ap_B'] - df_obs['Mag_Ap_G']
df_obs['Erro_Cor_B_G'] = np.sqrt(df_obs['Erro_Mag_B']**2 + df_obs['Erro_Mag_G']**2)

# Distância
df_obs['Distancia_pc'] = np.where(df_obs['Paralaxe_Gaia'] > 0, 1000.0 / df_obs['Paralaxe_Gaia'], np.nan)

# ======================================================================
# 6. Cálculo da Magnitude Absoluta e Propagação Correta de Erros
# ======================================================================
df_obs['Mag_Abs_G'] = df_obs['Mag_Ap_G'] - 5.0 * np.log10(df_obs['Distancia_pc']) + 5.0

# Propagação do erro da distância/paralaxe para a magnitude absoluta
erro_distancia_mag = (5.0 / np.log(10)) * (df_obs['Erro_Paralaxe_Gaia'] / df_obs['Paralaxe_Gaia'])
df_obs['Erro_Mag_Abs_G'] = np.sqrt(df_obs['Erro_Mag_G']**2 + erro_distancia_mag**2)

# Salvar catálogo completo
df_obs.to_csv('resultado_fotometria_calibrada_BG_Absoluta.csv', index=False)

# ======================================================================
# 7. Filtro de Qualidade Astrométrica (Plx / e_Plx > 5)
# ======================================================================
# Identificar o Alvo (LS 5039)
idx_alvo = np.argmin(coords_obs.separation(coord_alvo))

# Máscara para selecionar estrelas de campo com boa astrometria (excluindo o alvo)
# Só aceitamos relação Sinal-Ruído > 5 na paralaxe
mask_qualidade = (df_obs['Paralaxe_Gaia'] / df_obs['Erro_Paralaxe_Gaia']) > 5
mask_campo = mask_qualidade & (df_obs.index != idx_alvo)

df_campo_limpo = df_obs[mask_campo]
alvo_data = df_obs.iloc[idx_alvo]

print(f"Estrelas no campo após filtro de qualidade (S/N > 5): {len(df_campo_limpo)} de {len(df_obs)}")

# ======================================================================
# 8. Plotagem do Diagrama HR (Com Erros Propagados e Dados Filtrados)
# ======================================================================
plt.figure(figsize=(9, 7), dpi=100)

# Estrelas do Campo Geral (Apenas as aprovadas no filtro S/N > 5)
plt.errorbar(
    df_campo_limpo['Cor_B_G'], df_campo_limpo['Mag_Abs_G'], 
    xerr=df_campo_limpo['Erro_Cor_B_G'], yerr=df_campo_limpo['Erro_Mag_Abs_G'], 
    fmt='o', color='gray', ecolor='lightgray', 
    markersize=4, elinewidth=0.8, alpha=0.6, 
    markeredgecolor='black', label=r'Estrelas do Campo (Plx / e_Plx > 5)'
)

# Destaque Apenas do Alvo (LS 5039)
plt.errorbar(
    alvo_data['Cor_B_G'], alvo_data['Mag_Abs_G'], 
    xerr=alvo_data['Erro_Cor_B_G'], yerr=alvo_data['Erro_Mag_Abs_G'], 
    fmt='*', color='blue', ecolor='darkblue', 
    markersize=15, elinewidth=1.5, 
    markeredgecolor='black', zorder=5, label='LS 5039 (Alvo)'
)

# Formatação do Gráfico
plt.gca().invert_yaxis() 
plt.xlabel("Índice de Cor $(B - G)$ [mag]", fontsize=13)
plt.ylabel("Magnitude Absoluta $M_G$ [mag]", fontsize=13)
plt.title("Diagrama HR Observacional — Campo LS 5039", fontsize=14, pad=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('diagrama_cor_magnitude_BG_Absoluta_AltaQualidade.png', dpi=300)
plt.show()