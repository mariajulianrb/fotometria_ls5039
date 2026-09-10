import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

# Configurações de entrada e parâmetros da calibração
arq_in = 'fotometria_limpa_G.csv'
arq_out = 'resultado_fotometria_G_calibrada.csv'

zp_offset = 25.0
raio_match_arcsec = 2.0
num_top_estrelas = 500
mag_lim_gaia = (10.0, 18.0)
cut_sigma = 2.5

def reta(x, a, b):
    return a * x + b

# Leitura e ajuste da magnitude instrumental
df = pd.read_csv(arq_in)
df.columns = df.columns.str.strip()
df['Mag_Inst'] += zp_offset

bg_std = df['Std_Fundo_Local'].iloc[0]
ap_area = df['Area_Ap'].iloc[0]
t_exp = df['Exptime'].iloc[0]
col_erro = 'Erro_Mag' if 'Erro_Mag' in df.columns else 'Erro_Mag_Inst'

sky_obs = SkyCoord(ra=df['RA_deg'].values * u.deg, dec=df['Dec_deg'].values * u.deg)

# Consulta ao Gaia DR3 (banda G)
centro_campo = SkyCoord(ra=df['RA_deg'].mean() * u.deg, dec=df['Dec_deg'].mean() * u.deg)
viz = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag', 'e_Gmag'], row_limit=-1)
consulta = viz.query_region(centro_campo, radius=15.0 * u.arcmin, catalog='I/355/gaiadr3')[0]

cat_gaia = consulta[~np.isnan(consulta['Gmag'])]
sel_mag = (cat_gaia['Gmag'] > mag_lim_gaia[0]) & (cat_gaia['Gmag'] < mag_lim_gaia[1])
cat_gaia = cat_gaia[sel_mag]

sky_gaia = SkyCoord(ra=cat_gaia['RA_ICRS'].value * u.deg, dec=cat_gaia['DE_ICRS'].value * u.deg)

# Cross-match astrométrico
idx_cat, sep2d, _ = sky_obs.match_to_catalog_sky(sky_gaia)
m_validos = sep2d < (raio_match_arcsec * u.arcsec)

m_cat_vals = cat_gaia['Gmag'][idx_cat[m_validos]]
if hasattr(m_cat_vals, 'value'):
    m_cat_vals = m_cat_vals.value

df_matched = pd.DataFrame({
    'm_inst': df['Mag_Inst'].values[m_validos],
    'm_cat': m_cat_vals,
    'm_err': df[col_erro].values[m_validos]
}).sort_values('m_inst').reset_index(drop=True)

# Subconjunto das 500 mais brilhantes isolado para a regressão
df_fit = df_matched.head(num_top_estrelas)
x_fit = df_fit['m_inst'].values
y_fit = df_fit['m_cat'].values

# Regressão linear com corte estatístico em 2.5 sigma
p0, _ = curve_fit(reta, x_fit, y_fit)
residuos = y_fit - reta(x_fit, *p0)

clipped_res = sigma_clip(residuos, sigma=cut_sigma)
inliers = ~clipped_res.mask

popt, pcov = curve_fit(reta, x_fit[inliers], y_fit[inliers])
alfa, c = popt
e_alfa, e_c = np.sqrt(np.diag(pcov))

# Magnitude limite (5 sigma) e calibração de toda a tabela original
noise_ap = bg_std * np.sqrt(ap_area)
m_inst_5sig = -2.5 * np.log10((5.0 * noise_ap) / t_exp) + zp_offset
mag_limite_5sig = reta(m_inst_5sig, alfa, c)

df['Mag_G_Calibrada'] = reta(df['Mag_Inst'], alfa, c)
df.to_csv(arq_out, index=False)

print(f"Total de pares encontrados: {len(df_matched)}")
print(f"Amostra selecionada (mais brilhantes): {len(df_fit)}")
print(f"Fontes efetivamente usadas (inliers): {inliers.sum()}")
print(f"Ajuste: G = ({alfa:.4f} +/- {e_alfa:.4f}) * m_inst + ({c:.4f} +/- {e_c:.4f})")
print(f"Limite 5-sigma: {mag_limite_5sig:.2f} mag")

# ----------------------------------------------------------------------
# 1. Gráfico de Calibração Fotométrica
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 6), dpi=100)

plt.errorbar(
    df_matched['m_inst'], df_matched['m_cat'], xerr=df_matched['m_err'],
    fmt='.', color='darkgray', alpha=0.35, zorder=1, label='Todas as fontes pareadas'
)

if (~inliers).any():
    plt.errorbar(
        x_fit[~inliers], y_fit[~inliers], xerr=df_fit['m_err'].values[~inliers],
        fmt='x', color='crimson', alpha=0.8, zorder=2, label='Outliers descartados'
    )

plt.errorbar(
    x_fit[inliers], y_fit[inliers], xerr=df_fit['m_err'].values[inliers],
    fmt='o', color='royalblue', ecolor='darkred', elinewidth=1.2,
    capsize=2, alpha=0.9, markeredgecolor='k', zorder=3,
    label=f'Inliers do fit (Top {num_top_estrelas})'
)

x_line = np.linspace(df_matched['m_inst'].min(), max(df_matched['m_inst'].max(), m_inst_5sig), 100)
plt.plot(
    x_line, reta(x_line, alfa, c),
    color='darkorange', linestyle='--', linewidth=2, zorder=4,
    label=rf'$G = ({alfa:.3f} \pm {e_alfa:.3f}) \cdot m_{{inst}} + ({c:.2f} \pm {e_c:.2f})$'
)

plt.axvline(x=m_inst_5sig, color='purple', linestyle=':', linewidth=1.5, zorder=5)
plt.axhline(
    y=mag_limite_5sig, color='purple', linestyle=':', linewidth=1.5, zorder=5,
    label=f'Limite 5-$\sigma$ ({mag_limite_5sig:.2f} mag)'
)

plt.xlabel('Magnitude Instrumental ($ZP = 25$)', fontsize=11)
plt.ylabel('Magnitude Aparente Gaia DR3 ($G$)', fontsize=11)
plt.title('Calibração Fotométrica - Banda G (Gaia DR3)', fontsize=12, pad=10)
plt.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=9, loc='best')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('grafico_calibracao_G.png', dpi=300)
plt.show()

# ----------------------------------------------------------------------
# 2. Gráfico de Resíduos
# ----------------------------------------------------------------------
x_inliers = x_fit[inliers]
y_inliers = y_fit[inliers]
residuos_finais = y_inliers - reta(x_inliers, alfa, c)

plt.figure(figsize=(8, 5), dpi=100)
plt.scatter(
    y_inliers, 
    residuos_finais, 
    color='purple', alpha=0.6, edgecolor='k', linewidth=0.8, zorder=2
)
plt.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=1)

plt.xlabel("Magnitude Aparente Catálogo ($G$)")
plt.ylabel('Resíduo (Catálogo - Ajuste)')
plt.title('Resíduos do Ajuste Linear (Banda G)')
plt.grid(True, linestyle=':', alpha=0.6)

plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig('grafico_residuos_G.png', dpi=300)
plt.show()

# ----------------------------------------------------------------------
# 3. Histograma de Distribuição de Magnitudes
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 6), dpi=100)

plt.hist(
    df['Mag_G_Calibrada'].dropna(),
    bins=25,
    color='mediumseagreen',
    edgecolor='black',
    alpha=0.75,
    label='Fontes Extraídas'
)

plt.axvline(
    mag_limite_5sig, 
    color='darkred', 
    linestyle='--', 
    linewidth=2.5, 
    label=f'Limite 5-$\sigma$ ({mag_limite_5sig:.2f} mag)'
)

plt.xlabel('Magnitude Calibrada (G)', fontsize=12)
plt.ylabel('Número de Estrelas (N)', fontsize=12)
plt.title('Distribuição de Magnitudes e Profundidade (Banda G)', fontsize=13, pad=12)
plt.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('grafico_histograma_G.png', dpi=300)
plt.show()