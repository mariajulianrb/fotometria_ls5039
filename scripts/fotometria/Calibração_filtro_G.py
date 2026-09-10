import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

# Parametros de entrada e limites
arq_in = 'fotometria_limpa_G.csv'
arq_out = 'resultado_fotometria_G_calibrada.csv'

zp_base = 25.0
raio_match = 2.0  # arcsec
n_top = 500
lim_gaia = (10.0, 18.0)
sig_cut = 2.5

def reta(x, a, b):
    return a * x + b

# Leitura dos dados e ZP inicial
df = pd.read_csv(arq_in)
df.columns = df.columns.str.strip()
df['Mag_Inst'] += zp_base

std_bkg = df['Std_Fundo_Local'].iloc[0]
area_ap = df['Area_Ap'].iloc[0]
t_exp = df['Exptime'].iloc[0]
col_err = 'Erro_Mag' if 'Erro_Mag' in df.columns else 'Erro_Mag_Inst'

coords_obs = SkyCoord(ra=df['RA_deg'].values * u.deg, dec=df['Dec_deg'].values * u.deg)

# Consulta Gaia DR3 via VizieR
centro = SkyCoord(ra=df['RA_deg'].mean() * u.deg, dec=df['Dec_deg'].mean() * u.deg)
viz = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag', 'e_Gmag'], row_limit=-1)
consulta = viz.query_region(centro, radius=15.0 * u.arcmin, catalog='I/355/gaiadr3')[0]

cat = consulta[~np.isnan(consulta['Gmag'])]
cat = cat[(cat['Gmag'] > lim_gaia[0]) & (cat['Gmag'] < lim_gaia[1])]
coords_gaia = SkyCoord(ra=cat['RA_ICRS'].value * u.deg, dec=cat['DE_ICRS'].value * u.deg)

# Cross-match astrometric
idx, sep, _ = coords_obs.match_to_catalog_sky(coords_gaia)
mask = sep < (raio_match * u.arcsec)

g_cat = cat['Gmag'][idx[mask]]
if hasattr(g_cat, 'value'):
    g_cat = g_cat.value

df_match = pd.DataFrame({
    'm_inst': df['Mag_Inst'].values[mask],
    'm_cat': g_cat,
    'm_err': df[col_err].values[mask]
}).sort_values('m_inst').reset_index(drop=True)

# Selecao das N mais brilhantes para o fit
df_sub = df_match.head(n_top)
x_fit, y_fit = df_sub['m_inst'].values, df_sub['m_cat'].values

# Fit inicial e rejeicao de outliers
p_init, _ = curve_fit(reta, x_fit, y_fit)
residuos = y_fit - reta(x_fit, *p_init)
inliers = ~sigma_clip(residuos, sigma=sig_cut).mask

# Fit final
popt, pcov = curve_fit(reta, x_fit[inliers], y_fit[inliers])
alfa, c = popt
e_alfa, e_c = np.sqrt(np.diag(pcov))

# Magnitude limite 5-sigma
ruido = std_bkg * np.sqrt(area_ap)
m_inst_5sig = -2.5 * np.log10((5.0 * ruido) / t_exp) + zp_base
mag_lim_5sig = reta(m_inst_5sig, alfa, c)

# Exportacao dos resultados
df['Mag_G_Calibrada'] = reta(df['Mag_Inst'], alfa, c)
df.to_csv(arq_out, index=False)

print(f"Matchs totais: {len(df_match)} | Usados no fit: {inliers.sum()}/{len(df_sub)}")
print(f"Ajuste: G = ({alfa:.4f} +/- {e_alfa:.4f}) * m_inst + ({c:.4f} +/- {e_c:.4f})")
print(f"Limite 5-sigma: {mag_lim_5sig:.2f} mag")

# --- 1. Reta de Calibracao ---
plt.figure(figsize=(8, 5.5), dpi=100)

plt.errorbar(
    df_match['m_inst'], df_match['m_cat'], xerr=df_match['m_err'],
    fmt='.', color='darkgray', alpha=0.35, zorder=1, label='Fontes pareadas'
)

if (~inliers).any():
    plt.errorbar(
        x_fit[~inliers], y_fit[~inliers], xerr=df_sub['m_err'].values[~inliers],
        fmt='x', color='crimson', alpha=0.8, zorder=2, label='Outliers descartados'
    )

plt.errorbar(
    x_fit[inliers], y_fit[inliers], xerr=df_sub['m_err'].values[inliers],
    fmt='o', color='royalblue', ecolor='darkred', elinewidth=1.2,
    capsize=2, alpha=0.9, markeredgecolor='k', zorder=3,
    label=f'Inliers (Top {n_top})'
)

x_line = np.linspace(df_match['m_inst'].min(), max(df_match['m_inst'].max(), m_inst_5sig), 100)
plt.plot(
    x_line, reta(x_line, alfa, c),
    color='darkorange', linestyle='--', linewidth=2, zorder=4,
    label=rf'$G = ({alfa:.3f} \pm {e_alfa:.3f}) \cdot m_{{inst}} + ({c:.2f} \pm {e_c:.2f})$'
)



plt.xlabel('Magnitude Instrumental ($ZP = 25$)', fontsize=11)
plt.ylabel('Magnitude Gaia DR3 ($G$)', fontsize=11)
plt.title('Calibração Fotométrica - Banda G (Gaia DR3)', fontsize=12, pad=10)
plt.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=9, loc='best')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('grafico_calibracao_G.png', dpi=300)
plt.show()

# --- 2. Residuos ---
x_in, y_in = x_fit[inliers], y_fit[inliers]
res_finais = y_in - reta(x_in, alfa, c)

plt.figure(figsize=(7.5, 4.5), dpi=100)
plt.scatter(y_in, res_finais, color='purple', alpha=0.6, edgecolor='k', linewidth=0.8, zorder=2)
plt.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=1)

plt.xlabel("Magnitude Catálogo ($G$)")
plt.ylabel('Resíduo (Catálogo - Fit)')
plt.title('Resíduos da Calibração (Banda G)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig('grafico_residuos_G.png', dpi=300)
plt.show()

# --- 3. Histograma de Magnitudes ---
plt.figure(figsize=(8, 5.5), dpi=100)

plt.hist(
    df['Mag_G_Calibrada'].dropna(),
    bins=25, color='mediumseagreen', edgecolor='black', alpha=0.75, label='Fontes extraídas'
)

plt.axvline(
    mag_lim_5sig, color='darkred', linestyle='--', linewidth=2.5,
    label=f'Limite 5-$\sigma$ ({mag_lim_5sig:.2f} mag)'
)

plt.xlabel('Magnitude Calibrada (G)', fontsize=11)
plt.ylabel('Número de Estrelas (N)', fontsize=11)
plt.title('Distribuição de Magnitudes (Banda G)', fontsize=12, pad=10)
plt.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('grafico_histograma_G.png', dpi=300)
plt.show()