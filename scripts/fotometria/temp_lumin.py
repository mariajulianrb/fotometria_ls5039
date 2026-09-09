from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# ==========================================
# 1. ARQUIVOS & PARÂMETROS
# ==========================================
DIR_ATUAL = Path(__file__).resolve().parent
ARQUIVO_FOTOMETRIA = DIR_ATUAL / 'resultado_fotometria_calibrada_BG.csv'

RA_ALVO, DEC_ALVO = 276.5627, -14.8479  # Posição LS 5039
E_BG_MAX = 1.28
A_G_MAX = 3.07
DIST_ALVO_PC = 2500.0
MAG_BOL_SOL = 4.74

# Referência da Literatura para Comparação (Casares et al. 2005)
TEFF_LIT = 39000.0
LUM_LIT = 1.8e5

# Tabela Pecaut & Mamajek (2013)
bg0_grid = np.array([-0.363, -0.352, -0.330, -0.220, -0.110,  0.000,  0.165,  0.330])
teff_grid = np.array([42000,  39000,  31400,  20000,  14000,   9500,   8000,   7000])
bc_grid   = np.array([-3.80,  -3.50,  -3.00,  -2.10,  -1.20,  -0.30,   0.00,   0.05])

interp_teff = interp1d(bg0_grid, teff_grid, kind='linear', bounds_error=False, fill_value=(42000, 7000))
interp_bc   = interp1d(bg0_grid, bc_grid, kind='linear', bounds_error=False, fill_value=(-3.80, 0.05))

# ==========================================
# 2. LEITURA DOS DADOS & GAIA DR3
# ==========================================
df_obs = pd.read_csv(ARQUIVO_FOTOMETRIA)
coords_obs = SkyCoord(ra=df_obs['RA_deg'].values * u.deg, dec=df_obs['Dec_deg'].values * u.deg)

v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Plx', 'e_Plx'], row_limit=-1)
coord_centro = SkyCoord(ra=RA_ALVO * u.deg, dec=DEC_ALVO * u.deg)
tabela_gaia = v.query_region(coord_centro, radius=15 * u.arcmin, catalog='I/355/gaiadr3')[0].to_pandas()

# Filtro de qualidade na paralaxe
tabela_gaia = tabela_gaia[(tabela_gaia['Plx'] > 0) & (tabela_gaia['e_Plx'] / tabela_gaia['Plx'] < 0.20)]
coords_gaia = SkyCoord(ra=tabela_gaia['RA_ICRS'].values * u.deg, dec=tabela_gaia['DE_ICRS'].values * u.deg)

idx_gaia, d2d_gaia, _ = coords_obs.match_to_catalog_sky(coords_gaia)
valid_gaia = d2d_gaia < (2.0 * u.arcsec)

df = df_obs[valid_gaia].copy().reset_index(drop=True)
df['Paralaxe_mas'] = tabela_gaia['Plx'].iloc[idx_gaia[valid_gaia]].values

# ==========================================
# 3. CÁLCULO EMPÍRICO (TODAS AS ESTRELAS)
# ==========================================
df['Distancia_pc'] = 1000.0 / df['Paralaxe_mas']
modulo_distancia = 5 * np.log10(df['Distancia_pc']) - 5

fator_distancia = np.minimum(1.0, df['Distancia_pc'] / DIST_ALVO_PC)
df['E_BG_local'] = E_BG_MAX * fator_distancia
df['A_G_local'] = A_G_MAX * fator_distancia

# Derivação estritamente empírica dos parâmetros físicos
df['Cor_BG_0'] = df['Cor_B_G'] - df['E_BG_local']
df['Teff_emp'] = interp_teff(df['Cor_BG_0'])
df['BC_emp'] = interp_bc(df['Cor_BG_0'])

df['Mag_Abs_G0'] = df['Mag_Ap_G'] - modulo_distancia - df['A_G_local']
df['Mag_Bol'] = df['Mag_Abs_G0'] + df['BC_emp']
df['Luminosidade_Bol'] = 10 ** (-0.4 * (df['Mag_Bol'] - MAG_BOL_SOL))

# Identificação empírica da LS 5039 pelas coordenadas
coords_fisico = SkyCoord(ra=df['RA_deg'].values * u.deg, dec=df['Dec_deg'].values * u.deg)
idx_alvo = np.argmin(coords_fisico.separation(coord_centro))

ls5039_emp = df.iloc[idx_alvo]
df_campo = df.drop(index=idx_alvo).reset_index(drop=True)

# Filtro visual para o gráfico: estrelas quentes (Teff >= 8000 K)
df_campo = df_campo[df_campo['Teff_emp'] >= 8000].copy()

print(f"--- RESULTADO EMPÍRICO FOTOMÉTRICO (SUA REDUÇÃO) ---")
print(f"LS 5039 -> Teff: {ls5039_emp['Teff_emp']:.0f} K | L_bol: {ls5039_emp['Luminosidade_Bol']:.2e} L_sun")

# ==========================================
# 4. DIAGRAMA HR
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=120)

# 1. Estrelas do Campo
sc = ax.scatter(
    df_campo['Teff_emp'], df_campo['Luminosidade_Bol'],
    c=df_campo['Teff_emp'], cmap='RdYlBu_r',
    s=22, alpha=0.4, edgecolor='none', zorder=3,
    label=r'Campo ($T_{\mathrm{eff}} \geq 8000$ K)'
)

# 2. Ponto de Referência da Literatura
ax.scatter(
    TEFF_LIT, LUM_LIT,
    color='none', marker='o', s=200, edgecolor='black', linewidth=1.5, linestyle='--',
    zorder=8, label='Espectroscopia Literatura (Casares+05)'
)

# 3. Ponto Empírico Calculado do seu CSV
ax.scatter(
    ls5039_emp['Teff_emp'], ls5039_emp['Luminosidade_Bol'],
    color='cyan', marker='*', s=380, edgecolor='black', linewidth=1.2,
    zorder=10, label='LS 5039 (Sua Fotometria Empírica)'
)

# Anotação apontando para a sua estimativa empírica
ax.annotate(
    f'LS 5039 (Empírico)\n$T_{{eff}} \\approx {ls5039_emp["Teff_emp"]:.0f}$ K\n$L_{{bol}} \\approx {ls5039_emp["Luminosidade_Bol"]:.1e} L_\\odot$',
    xy=(ls5039_emp['Teff_emp'], ls5039_emp['Luminosidade_Bol']),
    xytext=(ls5039_emp['Teff_emp'] * 1.30, ls5039_emp['Luminosidade_Bol'] * 0.10),
    arrowprops=dict(facecolor='black', shrink=0.08, width=1.0, headwidth=5),
    fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
)

ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6, label=r'Luminosidade Solar ($1 L_\odot$)')

ax.set_yscale('log')
ax.set_xlim(44000, 7800)
ax.set_ylim(1e-1, 1e6)

ax.set_title('Diagrama HR — Estimativa Fotométrica Empírica vs. Literatura', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel(r'Temperatura Efetiva $T_{\mathrm{eff}}$ [K]', fontsize=11)
ax.set_ylabel(r'Luminosidade Bolométrica Total ($L_{\mathrm{bol}} / L_\odot$)', fontsize=11)

cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Temperatura Efetiva (K)', fontsize=10)

ax.grid(True, which='both', linestyle=':', alpha=0.3)
ax.legend(fontsize=9, loc='lower right', frameon=True)

plt.tight_layout()
out_fig = DIR_ATUAL / 'diagrama_HR_bolometrico_limpo.png'
plt.savefig(out_fig, dpi=300)
plt.show()