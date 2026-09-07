import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

# Configurações Iniciais
ARQUIVO_B = 'fotometria_bruta_G.csv'  # Arquivo da banda B/G
ARQUIVO_R = 'fotometria_bruta_R.csv'  # Arquivo da banda R
RA_ALVO, DEC_ALVO = 276.5627, -14.8479
RAIO_VIZIER = 15 * u.arcmin
TOLERANCIA_MATCH = 2.5 * u.arcsec

# Função para estimar a Temperatura Efetiva via Relação de Ballesteros (2012)
def calcular_temperatura(cor_B_V):
    return 4600 * (1 / (0.92 * cor_B_V + 1.7) + 1 / (0.92 * cor_B_V + 0.62))

# 1. Carregar os Dados e Definir Coordenadas
df_b = pd.read_csv(ARQUIVO_B)
df_r = pd.read_csv(ARQUIVO_R)
coord_alvo = SkyCoord(ra=RA_ALVO * u.deg, dec=DEC_ALVO * u.deg)

coords_b = SkyCoord(ra=df_b['RA_deg'].values * u.deg, dec=df_b['Dec_deg'].values * u.deg)
coords_r = SkyCoord(ra=df_r['RA_deg'].values * u.deg, dec=df_r['Dec_deg'].values * u.deg)

# 2. Cross-Match entre as Imagens (Filtro B vs R)
idx_r, d2d_br, _ = coords_b.match_to_catalog_sky(coords_r)
valid_br = d2d_br < TOLERANCIA_MATCH

df_obs = df_b[valid_br].copy().reset_index(drop=True)
df_obs.rename(columns={'Mag_Inst': 'Mag_Inst_B'}, inplace=True)
df_obs['Mag_Inst_R'] = df_r['Mag_Inst'].iloc[idx_r[valid_br]].values
coords_obs = coords_b[valid_br]

print(f"Estrelas combinadas (B + R): {len(df_obs)}")

# 3. Consulta ao Gaia DR3 no VizieR
print("Consultando o Gaia DR3 no VizieR...")
v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'BPmag', 'RPmag', 'Gmag'], row_limit=5000)
tabela_gaia = v.query_region(coord_alvo, radius=RAIO_VIZIER, catalog='I/355/gaiadr3')[0].to_pandas()
coords_gaia = SkyCoord(ra=tabela_gaia['RA_ICRS'].values * u.deg, dec=tabela_gaia['DE_ICRS'].values * u.deg)

# 4. Cross-Match com o Gaia para Calibração de Zero Point (ZP)
idx_gaia, d2d_gaia, _ = coords_obs.match_to_catalog_sky(coords_gaia)
valid_gaia = d2d_gaia < TOLERANCIA_MATCH

offset_B = tabela_gaia['BPmag'].iloc[idx_gaia[valid_gaia]].values - df_obs['Mag_Inst_B'][valid_gaia].values
offset_R = tabela_gaia['RPmag'].iloc[idx_gaia[valid_gaia]].values - df_obs['Mag_Inst_R'][valid_gaia].values

zp_B = np.nanmedian(offset_B)
zp_R = np.nanmedian(offset_R)
print(f"Zero Point B (BP): {zp_B:.3f} | Zero Point R (RP): {zp_R:.3f}")

# 5. Magnitudes Calibradas e Índice de Cor
df_obs['Mag_Ap_B'] = df_obs['Mag_Inst_B'] + zp_B
df_obs['Mag_Ap_R'] = df_obs['Mag_Inst_R'] + zp_R
df_obs['Cor_B_R'] = df_obs['Mag_Ap_B'] - df_obs['Mag_Ap_R']

# 6. Identificar o Alvo (LS 5039)
idx_alvo = np.argmin(coords_obs.separation(coord_alvo))
cor_obs_alvo = df_obs['Cor_B_R'].iloc[idx_alvo]
mag_R_obs_alvo = df_obs['Mag_Ap_R'].iloc[idx_alvo]

# 7. Cálculo da Extinção e Temperatura via Tipo Espectral de LS 5039 (O6.5V)
# Referência: Pecaut & Mamajek (2013) / Cardelli et al. (1989)
cor_intrinseca_BR_Ostar = -0.46  # (B - R)0 para estrela O6.5V
excesso_BR = cor_obs_alvo - cor_intrinseca_BR_Ostar

# Razões da Lei de Cardelli (CCM89): A_R = 1.30 * E(B-R) | A_B = 2.30 * E(B-R)
extincao_A_R = 1.30 * excesso_BR
extincao_A_B = 2.30 * excesso_BR

# Conversão empírica de (B - R) para (B - V) para a fórmula de Ballesteros
cor_obs_BV = 0.62 * cor_obs_alvo
cor_intrinseca_BV = -0.31

T_eff_aparente = calcular_temperatura(cor_obs_BV)
T_eff_real = calcular_temperatura(cor_intrinseca_BV)

# Aplicação da correção ao DataFrame completo (Desavermelhamento do campo)
df_obs['Mag_R_Intrinseca'] = df_obs['Mag_Ap_R'] - extincao_A_R
df_obs['Cor_BR_Intrinseco'] = df_obs['Cor_B_R'] - excesso_BR
df_obs.to_csv('resultado_fotometria_calibrada_BR.csv', index=False)

print("\n" + "=" * 55)
print("   RESULTADOS DA EXTINÇÃO E TEMPERATURA (LS 5039)")
print("=" * 55)
print(f"Cor Observada Medida (B - R):        {cor_obs_alvo:.3f} mag")
print(f"Cor Intrínseca Teórica (B - R)0:     {cor_intrinseca_BR_Ostar:.3f} mag (O6.5V)")
print(f"Excesso de Cor Derivado E(B - R):    {excesso_BR:.3f} mag")
print(f"Extinção Absoluta Banda R (A_R):     {extincao_A_R:.3f} mag")
print(f"Extinção Absoluta Banda B (A_B):     {extincao_A_B:.3f} mag")
print("-" * 55)
print(f"Temperatura Aparente (Sem Correção): {T_eff_aparente:.0f} K")
print(f"Temperatura Real (Com Correção):    {T_eff_real:.0f} K")
print("=" * 55 + "\n")

# 8. Plotagem do Diagrama HR (Aparente vs Corrigido)
fig, ax = plt.subplots(figsize=(9, 7))

# Estrelas do Campo Observadas
ax.scatter(df_obs['Cor_B_R'], df_obs['Mag_Ap_R'], c='black', alpha=0.4, s=25, label='Estrelas do Campo (Observado)')

# Alvo LS 5039 Observado (Com Poeira)
ax.scatter(cor_obs_alvo, mag_R_obs_alvo, c='red', s=180, marker='*', edgecolor='black', zorder=5, label='LS 5039 (Observado)')

# Alvo LS 5039 Intrinseco (Desavermelhado)
cor_int_alvo = cor_obs_alvo - excesso_BR
mag_int_alvo = mag_R_obs_alvo - extincao_A_R
ax.scatter(cor_int_alvo, mag_int_alvo, c='cyan', s=180, marker='*', edgecolor='black', zorder=5, label='LS 5039 (Intrínseco/Desavermelhado)')

# Vetor de Extinção
ax.annotate('', xy=(cor_int_alvo, mag_int_alvo), xytext=(cor_obs_alvo, mag_R_obs_alvo),
            arrowprops=dict(arrowstyle="->", color='blue', lw=2, ls='--'))

# Formatação do Gráfico
ax.invert_yaxis()
ax.set_xlabel('Índice de Cor $(B - R)$ [mag]', fontsize=13)
ax.set_ylabel('Magnitude $R$ [mag]', fontsize=13)
ax.set_title('Diagrama HR — Campo LS 5039 (Análise de Extinção Direta)', fontsize=14, pad=15)
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(fontsize=11, loc='best')

plt.tight_layout()
plt.savefig('diagrama_HR_BR_extincao_direta.png', dpi=300)
plt.show()