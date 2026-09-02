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
import warnings
from astropy.utils.exceptions import AstropyWarning

# Oculta os avisos poluentes do terminal
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=UserWarning)

def extrair_fontes_e_fotometria(imagem_fits, wcs_fits, r_ap=12.0):
    """Lê os pixels da imagem original e o WCS do arquivo do Astrometry."""
    data = fits.getdata(imagem_fits).astype(float)
    header_wcs = fits.getheader(wcs_fits)
    wcs = WCS(header_wcs)

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=4.0, threshold=5.0 * std)
    sources = daofind(data - median)

    if sources is None or len(sources) == 0:
        raise ValueError(f"Nenhuma fonte detectada no arquivo {imagem_fits}")

    posicoes = np.transpose((sources['x_centroid'], sources['y_centroid']))
    
    apertures = CircularAperture(posicoes, r=r_ap)
    annulus = CircularAnnulus(posicoes, r_in=r_ap + 5, r_out=r_ap + 15)
    
    sig_stats = ApertureStats(data, annulus)
    bkg_median = sig_stats.median
    
    phot = aperture_photometry(data, apertures)
    fluxo_liquido = phot['aperture_sum'] - (bkg_median * apertures.area)
    
    fluxo_liquido = np.maximum(fluxo_liquido, 1.0)
    m_inst = -2.5 * np.log10(fluxo_liquido)

    coords_sky = wcs.pixel_to_world(sources['x_centroid'], sources['y_centroid'])
    
    return coords_sky, m_inst, sources

print("--- Step 1: Extraindo fotometria das imagens ---")
sky_b, m_inst_b, _ = extrair_fontes_e_fotometria('calib_ls5039_b_001.fits', 'wcs_b.fits')
sky_r, m_inst_r, _ = extrair_fontes_e_fotometria('calib_ls5039_r_001.fits', 'wcs_r.fits')

print(f"Fontes detectadas: {len(m_inst_b)} (B) | {len(m_inst_r)} (R)")

# --- Step 2: Cruzando estrelas e buscando catálogo APASS ---
print("--- Step 2: Cruzando estrelas e buscando catálogo APASS ---")
idx_r, d2d, _ = sky_b.match_to_catalog_sky(sky_r)
mask_match = d2d < 10.0 * u.arcsec

m_inst_b_matched = m_inst_b[mask_match]
m_inst_r_matched = m_inst_r[idx_r[mask_match]]
sky_matched = sky_b[mask_match]

# --- Step 3: Obter Ponto Zero (ZP) ---
# CORRIGIDO: Nome da coluna do filtro vermelho no APASS é "r'mag"
v = Vizier(columns=['Bmag', "r'mag", 'RAJ2000', 'DEJ2000'])
resultado = v.query_region(sky_matched[0], radius=12 * u.arcmin, catalog='II/336/apass9')

if len(resultado) > 0:
    cat = resultado[0]
    cat_coords = SkyCoord(ra=cat['RAJ2000'], dec=cat['DEJ2000'], unit=(u.deg, u.deg))
    idx_cat, d2d_cat, _ = sky_matched.match_to_catalog_sky(cat_coords)
    mask_cat = d2d_cat < 2.0 * u.arcsec

    zp_b = np.nanmedian(cat['Bmag'][idx_cat[mask_cat]] - m_inst_b_matched[mask_cat])
    zp_r = np.nanmedian(cat["r'mag"][idx_cat[mask_cat]] - m_inst_r_matched[mask_cat])
else:
    zp_b, zp_r = 25.0, 25.0

m_ap_b = m_inst_b_matched + zp_b
m_ap_r = m_inst_r_matched + zp_r
indice_br = m_ap_b - m_ap_r

# --- Step 4: Diagrama CMD ---
print("--- Step 3: Gerando Diagrama e Resultados ---")
plt.figure(figsize=(9, 7))
plt.scatter(indice_br, m_ap_r, c='navy', alpha=0.5, s=25, label='Estrelas do Campo')

# Identificar LS 5039
ls5039_coord = SkyCoord(ra=276.5627*u.deg, dec=-14.8484*u.deg)
dist_ls = sky_matched.separation(ls5039_coord)
idx_ls = np.argmin(dist_ls)

if dist_ls[idx_ls] < 3.0 * u.arcsec:
    plt.scatter(indice_br[idx_ls], m_ap_r[idx_ls], color='red', s=120, edgecolors='black', label='LS 5039', zorder=5)

# --- Step 5: Gigantes Vermelhas ---

mask_gigantes = (indice_br > 1.5) & (m_ap_r < 14.0)
estrelas_gigantes_BR = indice_br[mask_gigantes]

if len(estrelas_gigantes_BR) > 0:
    plt.scatter(estrelas_gigantes_BR, m_ap_r[mask_gigantes], 
                color='orange', s=50, edgecolors='black', 
                label='Candidatas a Gigantes Vermelhas', zorder=4)

plt.gca().invert_yaxis()
plt.xlabel("Índice de Cor $(B - R)$ (mag)", fontsize=12)
plt.ylabel("Magnitude Aparente $R$ (mag)", fontsize=12)
plt.title("Diagrama Cor-Magnitude - Campo da LS 5039", fontsize=13)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("diagrama_cmd_final.png", dpi=300)

# --- Step 6: Cálculos Finais ---
BV_obs = 1.24
BV_intrinseco = -0.31
E_BV = BV_obs - BV_intrinseco
A_V = 3.1 * E_BV
BV_0 = BV_obs - E_BV
T_eff = 4600 * ((1 / (0.92 * BV_0 + 1.7)) + (1 / (0.92 * BV_0 + 0.62)))

print("\n=============================================")
print("  ESTRELA ALVO: LS 5039")
print("=============================================")
print(f"  Excesso de Cor E(B-V)      : {E_BV:.2f} mag")
print(f"  Extinção Total A_V         : {A_V:.2f} mag")
print(f"  Temperatura Efetiva (Teff) : {T_eff:.0f} K")

if len(estrelas_gigantes_BR) > 0:
    BR_obs_medio = np.nanmedian(estrelas_gigantes_BR)
    BR_intrinseco_gigante = 1.15 
    E_BR_campo = BR_obs_medio - BR_intrinseco_gigante
    A_V_campo = 2.3 * E_BR_campo
    
    print("\n=============================================")
    print("  EXTINÇÃO PELAS GIGANTES VERMELHAS (CAMPO)")
    print("=============================================")
    print(f"  Gigantes encontradas       : {len(estrelas_gigantes_BR)} estrelas")
    print(f"  Extinção Média (A_V)       : {A_V_campo:.2f} mag")
    print("=============================================\n")
else:
    print("\nNenhuma candidata a gigante vermelha detectada na região esperada.")