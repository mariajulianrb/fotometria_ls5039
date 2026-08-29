import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

def extrair_fontes_e_fotometria(fits_file, r_ap=12.0):
    """Lê o FITS com WCS, detecta estrelas e calcula a fotometria de abertura com subtração de céu."""
    with fits.open(fits_file) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
        wcs = WCS(header)

    # 1. Estimativa do fundo e detecção de estrelas
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=4.0, threshold=5.0 * std)
    sources = daofind(data - median)

    if sources is None or len(sources) == 0:
        raise ValueError(f"Nenhuma fonte detectada no arquivo {fits_file}")

    posicoes = np.transpose((sources['xcentroid'], sources['ycentroid']))
    
    # 2. Fotometria de Abertura e Anulus
    apertures = CircularAperture(posicoes, r=r_ap)
    annulus = CircularAnnulus(posicoes, r_in=r_ap + 5, r_out=r_ap + 15)
    
    sig_stats = ApertureStats(data, annulus)
    bkg_median = sig_stats.median
    
    phot = aperture_photometry(data, apertures)
    fluxo_liquido = phot['aperture_sum'] - (bkg_median * apertures.area)
    
    # 3. Magnitude instrumental
    fluxo_liquido = np.maximum(fluxo_liquido, 1.0) # Evita log negativo
    m_inst = -2.5 * np.log10(fluxo_liquido)

    # 4. Converte coordenadas de pixel (X, Y) para SkyCoord (RA, DEC)
    coords_sky = wcs.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
    
    return coords_sky, m_inst, sources

print("--- Step 1: Extraindo fotometria das imagens WCS ---")
sky_b, m_inst_b, _ = extrair_fontes_e_fotometria('wcs_b.fits')
sky_r, m_inst_r, _ = extrair_fontes_e_fotometria('wcs_r.fits')

print(f"Fontes detectadas: {len(m_inst_b)} (Filtro B) | {len(m_inst_r)} (Filtro R)")

# --- Step 2: Cruzamento espacial (Match) entre Filtro B e R ---
idx_b, idx_r, d2d, _ = sky_b.match_to_catalog_sky(sky_r)
mask_match = d2d < 1.5 * u.arcsec

m_inst_b_matched = m_inst_b[idx_b[mask_match]]
m_inst_r_matched = m_inst_r[mask_match]
sky_matched = sky_b[idx_b[mask_match]]

# --- Step 3: Obter Ponto Zero (ZP) consultando o Catálogo APASS ---
print("--- Step 2: Calibrando magnitudes com catálogo APASS via VizieR ---")
v = Vizier(columns=['Bmag', 'Rmag', 'RAJ2000', 'DEJ2000'])
resultado = v.query_region(sky_matched[0], radius=12 * u.arcmin, catalog='II/336/apass9')

if len(resultado) > 0:
    cat = resultado[0]
    cat_coords = SkyCoord(ra=cat['RAJ2000'], dec=cat['DEJ2000'], unit=(u.deg, u.deg))
    
    idx_obs, idx_cat, d2d_cat, _ = sky_matched.match_to_catalog_sky(cat_coords)
    mask_cat = d2d_cat < 2.0 * u.arcsec

    # ZP = Magnitude_Catalogo - Magnitude_Instrumental
    zp_b = np.nanmedian(cat['Bmag'][idx_cat[mask_cat]] - m_inst_b_matched[idx_obs[mask_cat]])
    zp_r = np.nanmedian(cat['Rmag'][idx_cat[mask_cat]] - m_inst_r_matched[idx_obs[mask_cat]])
    print(f"ZP Filtro B: {zp_b:.3f} | ZP Filtro R: {zp_r:.3f}")
else:
    print("⚠️ Catálogo indisponível. Usando ZP arbitrário = 25.0")
    zp_b, zp_r = 25.0, 25.0

# Magnitudes Aparentes finais
m_ap_b = m_inst_b_matched + zp_b
m_ap_r = m_inst_r_matched + zp_r
indice_br = m_ap_b - m_ap_r

# --- Step 4: Gerar o Diagrama Cor-Magnitude (CMD) ---
plt.figure(figsize=(9, 7))
plt.scatter(indice_br, m_ap_r, c='navy', alpha=0.6, s=30, label='Estrelas do Campo')

# Identificar a LS 5039 (RA: 276.5627, DEC: -14.8484)
ls5039_coord = SkyCoord(ra=276.5627*u.deg, dec=-14.8484*u.deg)
dist_ls = sky_matched.separation(ls5039_coord)
idx_ls = np.argmin(dist_ls)

if dist_ls[idx_ls] < 3.0 * u.arcsec:
    plt.scatter(indice_br[idx_ls], m_ap_r[idx_ls], color='red', s=120, edgecolors='black', label='LS 5039', zorder=5)

plt.gca().invert_yaxis()
plt.xlabel("Índice de Cor $(B - R)$ (mag)", fontsize=12)
plt.ylabel("Magnitude Aparente $R$ (mag)", fontsize=12)
plt.title("Diagrama Cor-Magnitude - Campo da LS 5039", fontsize=13)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("diagrama_cmd_ls5039.png", dpi=300)
print("✓ Diagrama salvo como 'diagrama_cmd_ls5039.png'")

# --- Step 5: Cálculo de Extinção e Temperatura para a LS 5039 ---
BV_obs = 1.24          # Catálogo SIMBAD
BV_intrinseco = -0.31   # O6.5V teórico

E_BV = BV_obs - BV_intrinseco
A_V = 3.1 * E_BV
BV_0 = BV_obs - E_BV
T_eff = 4600 * ((1 / (0.92 * BV_0 + 1.7)) + (1 / (0.92 * BV_0 + 0.62)))

print("\n=============================================")
print("  RESULTADOS DE EXTINÇÃO E TEMPERATURA")
print("=============================================")
print(f"  LS 5039 Magnitude R Aparente : {m_ap_r[idx_ls]:.2f} mag")
print(f"  Excesso de Cor E(B-V)        : {E_BV:.2f} mag")
print(f"  Extinção Total A_V           : {A_V:.2f} mag")
print(f"  Temperatura Efetiva (Teff)   : {T_eff:.0f} K")
print("=============================================")