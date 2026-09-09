import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

# ==========================================
# 1. CONFIGURAÇÕES & PARÂMETROS DA LITERATURA
# ==========================================
ARQUIVO_FOTOMETRIA = 'resultado_fotometria_calibrada_BG.csv'
RA_ALVO, DEC_ALVO = 276.5627, -14.8479  # Posição de LS 5039

# Valore de extinção e desavermelhamento (Casares et al. 2005; Megier et al. 2009)
E_BG = 1.51             # Excesso de cor E(B-G) no campo de LS 5039
A_G = 3.37              # Extinção total na banda G
MAG_ABS_SOL_G = 4.66    # Magnitude absoluta do Sol na banda G

def ballesteros_teff(cor_BV_0):
    """Calcula T_eff (K) usando a equação empírica de Ballesteros (2012)."""
    # Proteção contra a singularidade matemática em B-V = -0.673
    cor_BV_clean = np.maximum(cor_BV_0, -0.55)
    termo1 = 1.0 / (0.92 * cor_BV_clean + 1.7)
    termo2 = 1.0 / (0.92 * cor_BV_clean + 0.62)
    return 4600.0 * (termo1 + termo2)

# ==========================================
# 2. LEITURA DOS DADOS LOCAIS
# ==========================================
df_obs = pd.read_csv(ARQUIVO_FOTOMETRIA)
coords_obs = SkyCoord(ra=df_obs['RA_deg'].values * u.deg, dec=df_obs['Dec_deg'].values * u.deg)

# ==========================================
# 3. CONSULTA AO GAIA DR3 (PARALAXES E DISTÂNCIAS)
# ==========================================
print("Consultando Gaia DR3 no VizieR para resgatar paralaxes...")
v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Plx', 'e_Plx'], row_limit=-1)
coord_centro = SkyCoord(ra=RA_ALVO * u.deg, dec=DEC_ALVO * u.deg)
tabela_gaia = v.query_region(coord_centro, radius=15 * u.arcmin, catalog='I/355/gaiadr3')[0].to_pandas()

# Filtro de qualidade: Paralaxes válidas e com S/N > 3
tabela_gaia = tabela_gaia[(tabela_gaia['Plx'] > 0) & (tabela_gaia['Plx'] / tabela_gaia['e_Plx'] > 3)]
coords_gaia = SkyCoord(ra=tabela_gaia['RA_ICRS'].values * u.deg, dec=tabela_gaia['DE_ICRS'].values * u.deg)

# Cross-match astronômico
idx_gaia, d2d_gaia, _ = coords_obs.match_to_catalog_sky(coords_gaia)
valid_gaia = d2d_gaia < (2.5 * u.arcsec)

df_fisico = df_obs[valid_gaia].copy().reset_index(drop=True)
df_fisico['Paralaxe_mas'] = tabela_gaia['Plx'].iloc[idx_gaia[valid_gaia]].values
df_fisico['Erro_Paralaxe'] = tabela_gaia['e_Plx'].iloc[idx_gaia[valid_gaia]].values

# ==========================================
# 4. DESAVERMELHAMENTO, TEMPERATURA E LUMINOSIDADE
# ==========================================
# A. Correção do Índice de Cor Intrinseco (B - G)_0
df_fisico['Cor_BG_0'] = df_fisico['Cor_B_G'] - E_BG

# B. Conversão (B - G)_0 -> (B - V)_0 -> Temperatura Efetiva
df_fisico['Cor_BV_0'] = 0.86 * df_fisico['Cor_BG_0'] + 0.04
df_fisico['Teff'] = ballesteros_teff(df_fisico['Cor_BV_0'])

# C. Distância e Módulo de Distância
df_fisico['Distancia_pc'] = 1000.0 / df_fisico['Paralaxe_mas']
modulo_distancia = 5 * np.log10(df_fisico['Distancia_pc']) - 5

# D. Magnitude Absoluta Desavermelhada M_{G,0} e Luminosidade L/L_sun
df_fisico['Mag_Abs_G0'] = df_fisico['Mag_Ap_G'] - modulo_distancia - A_G
df_fisico['Luminosidade'] = 10 ** (-0.4 * (df_fisico['Mag_Abs_G0'] - MAG_ABS_SOL_G))

# E. Identificar a estrela alvo (LS 5039)
coords_fisico = SkyCoord(ra=df_fisico['RA_deg'].values * u.deg, dec=df_fisico['Dec_deg'].values * u.deg)
idx_alvo = np.argmin(coords_fisico.separation(coord_centro))

print(f"Total de estrelas processadas no Diagrama HR: {len(df_fisico)}")

# ==========================================
# 5. DIAGRAMA HR: TEMPERATURA VS LUMINOSIDADE
# ==========================================
plt.figure(figsize=(10, 8), dpi=100)

# Mapeamento de cores baseado na Temperatura
sc = plt.scatter(
    df_fisico['Teff'], df_fisico['Luminosidade'],
    c=df_fisico['Teff'], cmap='Spectral_r',
    s=30, alpha=0.75, edgecolor='k', linewidth=0.3,
    label='Estrelas do Campo (Corrigidas)'
)

# Destaque do Alvo (LS 5039)
plt.scatter(
    df_fisico['Teff'].iloc[idx_alvo], df_fisico['Luminosidade'].iloc[idx_alvo],
    color='cyan', marker='*', s=250, edgecolor='black', linewidth=1.2,
    zorder=5, label=f'LS 5039 ($T_{{eff}} \\approx {df_fisico["Teff"].iloc[idx_alvo]:.0f}$ K)'
)

# Linha de referência da Luminosidade Solar
plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, label='Luminosidade Solar ($1 L_\\odot$)')

# Escalas e Inversão padrão de Diagramas HR
plt.yscale('log')
plt.gca().invert_xaxis()  # Temperatura diminui da esquerda para a direita

# Formatação do Gráfico
plt.title(r'Diagrama HR Desavermelhado ($E(B-G)=1.51$, $A_G=3.37$)', fontsize=14, pad=15)
plt.xlabel(r'Temperatura Efetiva Estimada $T_{\mathrm{eff}}$ [K]', fontsize=12)
plt.ylabel(r'Luminosidade Intrínseca ($L / L_\odot$)', fontsize=12)

cbar = plt.colorbar(sc)
cbar.set_label('Temperatura Efetiva (K)', fontsize=11)

plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend(fontsize=11, loc='lower left')
plt.tight_layout()

plt.savefig('diagrama_HR_Temperatura_Luminosidade_Corrigido.png', dpi=300)
plt.show()