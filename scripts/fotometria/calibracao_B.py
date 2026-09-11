import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

arq_in = 'fotometria_limpa_B.csv'
arq_out = 'resultado_fotometria_B_calibrada.csv'

zp_base = 25.0
raio_match = 2.0  # arcsec
n_top = 500
lim_apass = (10.0, 18.0)
sig_cut = 2.5

def modelo_linear(x, a, b):
    return a * x + b

df = pd.read_csv(arq_in)
df.columns = df.columns.str.strip()
df['Mag_Inst'] += zp_base

std_bkg = df['Std_Fundo_Local'].iloc[0]
area_ap = df['Area_Ap'].iloc[0]
t_exp = df['Exptime'].iloc[0]
col_err = 'Erro_Mag' if 'Erro_Mag' in df.columns else 'Erro_Mag_Inst'

coords_obs = SkyCoord(ra=df['RA_deg'].values * u.deg, dec=df['Dec_deg'].values * u.deg)

centro = SkyCoord(ra=df['RA_deg'].mean() * u.deg, dec=df['Dec_deg'].mean() * u.deg)
v = Vizier(columns=['RAJ2000', 'DEJ2000', 'Bmag', 'e_Bmag'], row_limit=-1)
cat_raw = v.query_region(centro, radius=15.0 * u.arcmin, catalog='II/336/apass9')[0]

cat = cat_raw[~np.isnan(cat_raw['Bmag'])]
cat = cat[(cat['Bmag'] > lim_apass[0]) & (cat['Bmag'] < lim_apass[1])]
coords_apass = SkyCoord(ra=cat['RAJ2000'].value * u.deg, dec=cat['DEJ2000'].value * u.deg)

idx, sep, _ = coords_obs.match_to_catalog_sky(coords_apass)
mask_match = sep < (raio_match * u.arcsec)

b_cat = cat['Bmag'][idx[mask_match]]
if hasattr(b_cat, 'value'):
    b_cat = b_cat.value

df_match = pd.DataFrame({
    'm_inst': df['Mag_Inst'].values[mask_match],
    'm_cat': b_cat,
    'm_err': df[col_err].values[mask_match]
}).sort_values('m_inst').reset_index(drop=True)

df_sub = df_match.head(n_top)
x_fit = df_sub['m_inst'].values
y_fit = df_sub['m_cat'].values

p_init, _ = curve_fit(modelo_linear, x_fit, y_fit)
res_init = y_fit - modelo_linear(x_fit, *p_init)

clip = sigma_clip(res_init, sigma=sig_cut)
inliers = ~clip.mask

popt, pcov = curve_fit(modelo_linear, x_fit[inliers], y_fit[inliers])
alfa, c = popt
err_alfa, err_c = np.sqrt(np.diag(pcov))

ruido_ap = std_bkg * np.sqrt(area_ap)
m_inst_5sig = -2.5 * np.log10((5.0 * ruido_ap) / t_exp) + zp_base
mag_lim_5sig = modelo_linear(m_inst_5sig, alfa, c)

err_mag_lim_5sig = np.sqrt(
    (m_inst_5sig ** 2) * pcov[0, 0] + pcov[1, 1] + 2 * m_inst_5sig * pcov[0, 1]
)

df['Mag_B_Calibrada'] = modelo_linear(df['Mag_Inst'], alfa, c)
df.to_csv(arq_out, index=False)

print(f"Matchs totais: {len(df_match)}")
print(f"Fontes no fit (top {n_top}): {len(df_sub)}")
print(f"Inliers finais: {inliers.sum()}")
print(f"Reta: B = ({alfa:.4f} +/- {err_alfa:.4f}) * m_inst + ({c:.4f} +/- {err_c:.4f})")
print(f"Limite 5-sigma: {mag_lim_5sig:.2f} +/- {err_mag_lim_5sig:.2f} mag")

plt.figure(figsize=(8, 5.5))

if (~inliers).any():
    plt.errorbar(
        x_fit[~inliers], y_fit[~inliers], xerr=df_sub['m_err'].values[~inliers],
        fmt='x', color='red', alpha=0.7, label='Outliers'
    )

plt.errorbar(
    x_fit[inliers], y_fit[inliers], xerr=df_sub['m_err'].values[inliers],
    fmt='o', color='navy', ecolor='darkred', elinewidth=1,
    capsize=2, alpha=0.8, markeredgecolor='k', label=f'Inliers({n_top})'
)

x_plot = np.linspace(df_match['m_inst'].min(), max(df_match['m_inst'].max(), m_inst_5sig), 100)
plt.plot(
    x_plot, modelo_linear(x_plot, alfa, c),
    color='darkorange', linestyle='--', linewidth=1.8,
    label=rf'$B = ({alfa:.3f} \pm {err_alfa:.3f}) \cdot m_{{inst}} + ({c:.2f} \pm {err_c:.2f})$'
)

plt.xlabel('Magnitude Instrumental')
plt.ylabel('Magnitude APASS DR9 ($B$)')
plt.title('Calibração Fotométrica — Banda B')
plt.legend(loc='best', frameon=True)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('grafico_calibracao_B.png', dpi=300)
plt.show()

x_in = x_fit[inliers]
y_in = y_fit[inliers]
res_finais = y_in - modelo_linear(x_in, alfa, c)

plt.figure(figsize=(7.5, 4.5))
plt.scatter(y_in, res_finais, color='indigo', alpha=0.6, edgecolor='k', linewidth=0.5)
plt.axhline(0, color='k', linestyle='--', linewidth=1)

plt.xlabel("Magnitude APASS ($B$)")
plt.ylabel('Resíduo (Catálogo - Ajuste)')
plt.title('Resíduos do Ajuste Linear — Banda B')
plt.grid(True, linestyle=':', alpha=0.5)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig('grafico_residuos_B.png', dpi=300)
plt.show()

plt.figure(figsize=(7.5, 5))
plt.hist(
    df['Mag_B_Calibrada'].dropna(),
    bins=25, color='seagreen', edgecolor='black', alpha=0.7, label='Fontes extraídas'
)

plt.axvline(
    mag_lim_5sig, color='crimson', linestyle='--', linewidth=2,
    label=rf'Limite 5$\sigma$ ({mag_lim_5sig:.2f} $\pm$ {err_mag_lim_5sig:.2f} mag)'
)

plt.xlabel('Magnitude Calibrada ($B$)')
plt.ylabel('Número de Estrelas ($N$)')
plt.title('Distribuição de Magnitudes — Banda B')
plt.legend(frameon=True)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('grafico_histograma_B.png', dpi=300)
plt.show()