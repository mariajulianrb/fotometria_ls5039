import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

# Configurações Iniciais
ARQUIVO_B = 'fotometria_bruta_G.csv'  # Filtro B (ou G)
ARQUIVO_R = 'fotometria_bruta_R.csv'  # Filtro R
RA_ALVO, DEC_ALVO = 276.5627, -14.8479
RAIO_VIZIER = 15 * u.arcmin
TOLERANCIA_MATCH = 2.5 * u.arcsec

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
v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'BPmag', 'RPmag', 'Gmag', 'Plx'], row_limit=5000)
tabela_gaia = v.query_region(coord_alvo, radius=RAIO_VIZIER, catalog='I/355/gaiadr3')[0].to_pandas()
coords_gaia = SkyCoord(ra=tabela_gaia['RA_ICRS'].values * u.deg, dec=tabela_gaia['DE_ICRS'].values * u.deg)

# 4. Cross-Match com o Gaia para Calibração de Ponto Zero (ZP)
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
df_obs.to_csv('resultado_fotometria_calibrada_BR.csv', index=False)

# 6. Identificar o Alvo (LS 5039)
idx_alvo = np.argmin(coords_obs.separation(coord_alvo))

# 7. Cálculo da Extinção Interstelar via Gigantes Vermelhas (Red Clump)
mascara_rc = (
    (df_obs['Cor_B_R'] >= 1.4) & (df_obs['Cor_B_R'] <= 2.0) &
    (df_obs['Mag_Ap_R'] >= 11.0) & (df_obs['Mag_Ap_R'] <= 13.0)
)
df_rc = df_obs[mascara_rc]

if len(df_rc) > 0:
    cor_obs_rc = df_rc['Cor_B_R'].median()
    cor_intrinseca_rc = 1.25  # Valor teórico (B-R)0 para Gigantes K0III/Red Clump
    excesso_cor = cor_obs_rc - cor_intrinseca_rc
    extincao_A_R = 2.3 * excesso_cor  # R_R = 2.3 na Lei de Extinção de Cardelli

    print("\n" + "=" * 40)
    print("      RESULTADOS DA EXTINÇÃO INTERESTELAR")
    print("=" * 40)
    print(f"Gigantes Vermelhas isoladas: {len(df_rc)}")
    print(f"Cor Observada Mediana (B-R): {cor_obs_rc:.3f} mag")
    print(f"Excesso de Cor E(B-R): {excesso_cor:.3f} mag")
    print(f"Extinção Absoluta na Banda R (A_R): {extincao_A_R:.3f} mag")
    print("=" * 40 + "\n")
else:
    print("\nAviso: Nenhuma gigante vermelha encontrada na caixa do Red Clump.\n")

# 8. Plotagem do Diagrama HR Completo
plt.figure(figsize=(9, 7))

# Estrelas do Campo Geral
plt.scatter(df_obs['Cor_B_R'], df_obs['Mag_Ap_R'], c='black', alpha=0.5, s=25, label='Estrelas do Campo')

# Destaque das Gigantes Vermelhas (Red Clump) usadas para Extinção
if len(df_rc) > 0:
    plt.scatter(df_rc['Cor_B_R'], df_rc['Mag_Ap_R'], c='orange', s=45, marker='o', 
                edgecolor='darkorange', label='Gigantes Vermelhas (Red Clump)')

# Destaque do Alvo (LS 5039)
plt.scatter(df_obs['Cor_B_R'].iloc[idx_alvo], df_obs['Mag_Ap_R'].iloc[idx_alvo], 
            c='red', s=200, marker='*', edgecolor='black', zorder=5, label='LS 5039 (Alvo)')

# Ajuste da Linha de Tendência da Sequência Principal
mascara_tendencia = (df_obs['Cor_B_R'] > 0.6) & (df_obs['Cor_B_R'] < 1.8)
cor_limpa = df_obs['Cor_B_R'][mascara_tendencia]
mag_limpa = df_obs['Mag_Ap_R'][mascara_tendencia]

coeficientes = np.polyfit(cor_limpa, mag_limpa, 2)
polinomio = np.poly1d(coeficientes)

x_linha = np.linspace(0.6, 2.0, 100)
y_linha = polinomio(x_linha)

#plt.plot(x_linha, y_linha, color='dodgerblue', linestyle='--', linewidth=2.5, label='Sequência Principal')

# Formatação do Gráfico
plt.gca().invert_yaxis()  # Magnitudes mais brilhantes para cima
plt.xlabel('Índice de Cor $(B - R)$ [mag]', fontsize=13)
plt.ylabel('Magnitude Aparente $R$ [mag]', fontsize=13)
plt.title('Diagrama HR — Campo LS 5039 (com Análise de Extinção)', fontsize=14, pad=15)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('diagrama_HR_BR_extincao.png', dpi=300)
plt.show()