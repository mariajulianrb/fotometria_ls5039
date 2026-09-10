import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier
from astropy.stats import sigma_clip

# Parametros de entrada
arq_b = 'resultado_fotometria_B_calibrada.csv'
arq_r = 'resultado_fotometria_R_calibrada.csv'
arq_out = 'resultado_fotometria_calibrada_BR_Aparente.csv'

ra_alvo, dec_alvo = 276.5627, -14.8479
raio_vizier = 15.0  # arcmin
raio_match = 2.5    # arcsec
zp_base = 25.0

# Leitura e aplicacao do ZP inicial
df_b = pd.read_csv(arq_b)
df_r = pd.read_csv(arq_r)
df_b.columns, df_r.columns = df_b.columns.str.strip(), df_r.columns.str.strip()

df_b['Mag_Inst'] += zp_base
df_r['Mag_Inst'] += zp_base

coord_alvo = SkyCoord(ra=ra_alvo * u.deg, dec=dec_alvo * u.deg)
coords_b = SkyCoord(ra=df_b['RA_deg'].values * u.deg, dec=df_b['Dec_deg'].values * u.deg)
coords_r = SkyCoord(ra=df_r['RA_deg'].values * u.deg, dec=df_r['Dec_deg'].values * u.deg)

# Cross-match das bandas B e R
idx_r, sep, _ = coords_b.match_to_catalog_sky(coords_r)
mask_br = sep < (raio_match * u.arcsec)

df_obs = df_b[mask_br].copy().reset_index(drop=True)
df_obs.rename(columns={'Mag_Inst': 'Mag_Inst_B', 'Erro_Mag': 'Erro_Mag_B'}, inplace=True)

df_obs['Mag_Inst_R'] = df_r['Mag_Inst'].iloc[idx_r[mask_br]].values
col_err_r = 'Erro_Mag' if 'Erro_Mag' in df_r.columns else 'Erro_Mag_Inst'
df_obs['Erro_Mag_R'] = df_r[col_err_r].iloc[idx_r[mask_br]].values

coords_obs = coords_b[mask_br]

# Consulta APASS DR9 via VizieR para B e R (r'mag)
v_apass = Vizier(columns=['RAJ2000', 'DEJ2000', 'Bmag', "r'mag"], row_limit=-1)
cat_apass = v_apass.query_region(coord_alvo, radius=raio_vizier * u.arcmin, catalog='II/336/apass9')[0].to_pandas()

# Identificacao dinamica da coluna r do APASS (evita erros de apóstrofo/nome)
col_r_apass = [c for c in cat_apass.columns if 'r' in c.lower() and 'mag' in c.lower() and 'e_' not in c.lower()][0]

cat_apass = cat_apass.dropna(subset=['Bmag', col_r_apass])
coords_apass = SkyCoord(ra=cat_apass['RAJ2000'].values * u.deg, dec=cat_apass['DEJ2000'].values * u.deg)

# Calibracao de Zero Point residual com APASS
idx_ap, sep_ap, _ = coords_obs.match_to_catalog_sky(coords_apass)
mask_ap = sep_ap < (raio_match * u.arcsec)

off_b = cat_apass['Bmag'].iloc[idx_ap[mask_ap]].values - df_obs['Mag_Inst_B'][mask_ap].values
zp_b = np.ma.median(sigma_clip(off_b, sigma=2.5))

off_r = cat_apass[col_r_apass].iloc[idx_ap[mask_ap]].values - df_obs['Mag_Inst_R'][mask_ap].values
zp_r = np.ma.median(sigma_clip(off_r, sigma=2.5))

# Magnitudes aparentes e propagacao de erro na cor
df_obs['Mag_Ap_B'] = df_obs['Mag_Inst_B'] + zp_b
df_obs['Mag_Ap_R'] = df_obs['Mag_Inst_R'] + zp_r
df_obs['Cor_B_R'] = df_obs['Mag_Ap_B'] - df_obs['Mag_Ap_R']
df_obs['Erro_Cor_B_R'] = np.sqrt(df_obs['Erro_Mag_B']**2 + df_obs['Erro_Mag_R']**2)

df_obs.to_csv(arq_out, index=False)

# Identificacao do alvo
idx_alvo = np.argmin(coords_obs.separation(coord_alvo))

print(f"Fontes pareadas (B+R): {len(df_obs)}")
print(f"ZP Residual B (APASS): {zp_b:.3f} | ZP Residual R (APASS): {zp_r:.3f}")

# --- Diagrama Cor-Magnitude ---
plt.figure(figsize=(8, 6), dpi=100)

plt.errorbar(
    df_obs['Cor_B_R'], df_obs['Mag_Ap_R'], 
    xerr=df_obs['Erro_Cor_B_R'], yerr=df_obs['Erro_Mag_R'], 
    fmt='.', color='gray', alpha=0.5, elinewidth=0.8, zorder=1, label='Estrelas do campo'
)

plt.errorbar(
    df_obs['Cor_B_R'].iloc[idx_alvo], df_obs['Mag_Ap_R'].iloc[idx_alvo], 
    xerr=df_obs['Erro_Cor_B_R'].iloc[idx_alvo], yerr=df_obs['Erro_Mag_R'].iloc[idx_alvo], 
    fmt='*', color='crimson', ecolor='darkred', markersize=14, elinewidth=1.5, 
    markeredgecolor='k', zorder=5, label='LS 5039'
)

plt.gca().invert_yaxis()
plt.xlabel("Índice de Cor ($B - R$)")
plt.ylabel("Magnitude Aparente ($R$)")
plt.title("Diagrama Cor-Magnitude — Campo LS 5039")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True, loc='best')

plt.tight_layout()
plt.savefig('diagrama_cor_magnitude_BR.png', dpi=300)
plt.show()