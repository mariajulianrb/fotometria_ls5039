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
ARQUIVO_R = os.path.join(diretorio_atual, 'resultado_fotometria_R_calibrada.csv') 

RA_ALVO, DEC_ALVO = 276.5627, -14.8479
RAIO_VIZIER = 15 * u.arcmin
TOLERANCIA_MATCH = 2.5 * u.arcsec
ZP_OFFSET = 25.0

# ======================================================================
# 1. Carregar os Dados e Definir Coordenadas
# ======================================================================
df_b = pd.read_csv(ARQUIVO_B)
df_r = pd.read_csv(ARQUIVO_R)

df_b.columns = df_b.columns.str.strip()
df_r.columns = df_r.columns.str.strip()

# Aplicar o Zero Point base (ZP = 25)
df_b['Mag_Inst'] = df_b['Mag_Inst'] + ZP_OFFSET
df_r['Mag_Inst'] = df_r['Mag_Inst'] + ZP_OFFSET

coord_alvo = SkyCoord(ra=RA_ALVO * u.deg, dec=DEC_ALVO * u.deg)

coords_b = SkyCoord(ra=df_b['RA_deg'].values * u.deg, dec=df_b['Dec_deg'].values * u.deg)
coords_r = SkyCoord(ra=df_r['RA_deg'].values * u.deg, dec=df_r['Dec_deg'].values * u.deg)

# ======================================================================
# 2. Cross-Match entre as Imagens (Filtro B vs R) e Captura de Erros
# ======================================================================
idx_r, d2d_br, _ = coords_b.match_to_catalog_sky(coords_r)
valid_br = d2d_br < TOLERANCIA_MATCH

df_obs = df_b[valid_br].copy().reset_index(drop=True)
df_obs.rename(columns={'Mag_Inst': 'Mag_Inst_B', 'Erro_Mag': 'Erro_Mag_B'}, inplace=True)

df_obs['Mag_Inst_R'] = df_r['Mag_Inst'].iloc[idx_r[valid_br]].values
df_obs['Erro_Mag_R'] = df_r['Erro_Mag'].iloc[idx_r[valid_br]].values
coords_obs = coords_b[valid_br]

print(f"Estrelas combinadas (B + R): {len(df_obs)}")

# ======================================================================
# 3. Consulta ao APASS DR9 no VizieR (Referência Fotométrica)
# ======================================================================
print("Consultando APASS DR9 no VizieR para calibração de Ponto Zero...")
v_apass = Vizier(columns=['RAJ2000', 'DEJ2000', 'Bmag', "r'mag"], row_limit=-1)
tabela_apass = v_apass.query_region(coord_alvo, radius=RAIO_VIZIER, catalog='II/336/apass9')[0].to_pandas()
tabela_apass = tabela_apass.dropna(subset=['Bmag', "r'mag"]) 
coords_apass = SkyCoord(ra=tabela_apass['RAJ2000'].values * u.deg, dec=tabela_apass['DEJ2000'].values * u.deg)

# ======================================================================
# 4. Cross-Match e Calibração dos Pontos Zeros
# ======================================================================
idx_apass, d2d_apass, _ = coords_obs.match_to_catalog_sky(coords_apass)
valid_apass = d2d_apass < TOLERANCIA_MATCH

offset_B_bruto = tabela_apass['Bmag'].iloc[idx_apass[valid_apass]].values - df_obs['Mag_Inst_B'][valid_apass].values
offset_R_bruto = tabela_apass["r'mag"].iloc[idx_apass[valid_apass]].values - df_obs['Mag_Inst_R'][valid_apass].values

offset_B_limpo = sigma_clip(offset_B_bruto, sigma=2.5)
offset_R_limpo = sigma_clip(offset_R_bruto, sigma=2.5)

zp_B_residual = np.ma.median(offset_B_limpo)
zp_R_residual = np.ma.median(offset_R_limpo)
print(f"Residual Zero Point B (APASS B): {zp_B_residual:.3f} | Residual Zero Point R (APASS r'): {zp_R_residual:.3f}")

# ======================================================================
# 5. Magnitudes Aparentes Calibradas e Propagação de Erro na Cor
# ======================================================================
df_obs['Mag_Ap_B'] = df_obs['Mag_Inst_B'] + zp_B_residual
df_obs['Mag_Ap_R'] = df_obs['Mag_Inst_R'] + zp_R_residual
df_obs['Cor_B_R'] = df_obs['Mag_Ap_B'] - df_obs['Mag_Ap_R']

# O erro da cor é a soma em quadratura dos erros fotométricos individuais
df_obs['Erro_Cor_B_R'] = np.sqrt(df_obs['Erro_Mag_B']**2 + df_obs['Erro_Mag_R']**2)

# ======================================================================
# 6. Identificar o Alvo (LS 5039) e Salvar
# ======================================================================
idx_alvo = np.argmin(coords_obs.separation(coord_alvo))

df_obs.to_csv('resultado_fotometria_calibrada_BR_Aparente.csv', index=False)
print("Catálogo de magnitudes aparentes salvo com sucesso.")

# ======================================================================
# 7. Plotagem do Diagrama de Cor-Magnitude (Aparente)
# ======================================================================
plt.figure(figsize=(9, 7), dpi=100)

# Estrelas do Campo Geral (Usando Magnitude Aparente)
plt.errorbar(
    df_obs['Cor_B_R'], df_obs['Mag_Ap_R'], 
    xerr=df_obs['Erro_Cor_B_R'], yerr=df_obs['Erro_Mag_R'], 
    fmt='o', color='gray', ecolor='lightgray', 
    markersize=4, elinewidth=0.8, alpha=0.6, 
    markeredgecolor='black', label='Estrelas do Campo'
)

# Destaque Apenas do Alvo (LS 5039)
plt.errorbar(
    df_obs['Cor_B_R'].iloc[idx_alvo], df_obs['Mag_Ap_R'].iloc[idx_alvo], 
    xerr=df_obs['Erro_Cor_B_R'].iloc[idx_alvo], yerr=df_obs['Erro_Mag_R'].iloc[idx_alvo], 
    fmt='*', color='red', ecolor='darkred', 
    markersize=15, elinewidth=1.5, 
    markeredgecolor='black', zorder=5, label='LS 5039 (Alvo)'
)

# Formatação do Gráfico
plt.gca().invert_yaxis() # Estrelas mais brilhantes (menor magnitude) no topo
plt.xlabel("Índice de Cor (B - r') [mag]", fontsize=13)
plt.ylabel("Magnitude Aparente m_R [mag]", fontsize=13)
plt.title("Diagrama Cor-Magnitude Observacional — Campo LS 5039", fontsize=14, pad=15)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('diagrama_cor_magnitude_B-R_Aparente.png', dpi=300)
plt.show()