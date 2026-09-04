import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

# ==========================================
# CONFIGURAÇÕES INICIAIS
# ==========================================
arquivo_imagem = '/home/maju/Downloads/dados/astronometry/LS5039_B_wcs.fits'
raio_abertura = 5.0 # Raio em pixels para medir a luz da estrela

# Abre a imagem e extrai dados e WCS
hdul = fits.open(arquivo_imagem)
image_data = hdul[0].data
header = hdul[0].header
wcs = WCS(header)

# ==========================================
# 1. RUÍDO DE FUNDO E DETECÇÃO
# ==========================================
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)

# Detecta estrelas com picos 5x acima do ruído de fundo (5-sigma)
daofind = DAOStarFinder(fwhm=3.0, threshold=5. * std)
fontes = daofind(image_data - median)
print(f"Estrelas detectadas na imagem: {len(fontes)}")

# ==========================================
# 2. FOTOMETRIA INSTRUMENTAL E WCS
# ==========================================
# Cria as aberturas circulares nas posições (X, Y) detectadas
posicoes = np.transpose((fontes['xcentroid'], fontes['ycentroid']))
aberturas = CircularAperture(posicoes, r=raio_abertura)

# Soma a luz (fluxo) dentro de cada abertura
tabela_fotometria = aperture_photometry(image_data - median, aberturas)

# Calcula a magnitude instrumental: m = -2.5 * log10(fluxo)
# Filtramos fluxos negativos para evitar erros matemáticos
fluxos_validos = tabela_fotometria['aperture_sum'] > 0
tabela_filtrada = tabela_fotometria[fluxos_validos]
fluxo = tabela_filtrada['aperture_sum']

tabela_filtrada['mag_inst'] = -2.5 * np.log10(fluxo)

# Converte o (X, Y) dessas estrelas válidas para RA e Dec
coords_imagem = wcs.pixel_to_world(tabela_filtrada['xcenter'], tabela_filtrada['ycenter'])

# ==========================================
# 3. BUSCA NO CATÁLOGO (VIZIER - UCAC4)
# ==========================================
# Pegamos o centro da imagem e o tamanho aproximado para buscar no catálogo
centro_ra, centro_dec = wcs.pixel_to_world(image_data.shape[1]/2, image_data.shape[0]/2).ra.deg, wcs.pixel_to_world(image_data.shape[1]/2, image_data.shape[0]/2).dec.deg
centro_coord = SkyCoord(ra=centro_ra, dec=centro_dec, unit=(u.deg, u.deg))

# Busca catálogo UCAC4 (banda V) num raio de 15 arcmin (ajuste conforme o FOV do seu telescópio)
vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'Vmag'])
vizier.ROW_LIMIT = -1 # Sem limite de linhas
catalogo = vizier.query_region(centro_coord, radius=15*u.arcmin, catalog='I/322A/out')[0]

# Remove entradas sem magnitude V medida
catalogo = catalogo[~np.isnan(catalogo['Vmag'])]
coords_catalogo = SkyCoord(ra=catalogo['RAJ2000'], dec=catalogo['DEJ2000'], unit=(u.deg, u.deg))

# ==========================================
# 4. CROSS-MATCHING (CRUZAMENTO)
# ==========================================
# Encontra a estrela do catálogo mais próxima para cada estrela da nossa imagem
idx_catalogo, d2d, d3d = coords_imagem.match_to_catalog_sky(coords_catalogo)

# Mantém apenas os pares que estão muito próximos (ex: menos de 2 arcsec de distância)
# Isso garante que não estamos pareando estrelas diferentes que calharam de estar na mesma área
limite_distancia = 2.0 * u.arcsec
pares_validos = d2d < limite_distancia

mag_inst_pareada = tabela_filtrada['mag_inst'][pares_validos]
mag_aparente_pareada = catalogo['Vmag'][idx_catalogo[pares_validos]]

print(f"Pares perfeitos encontrados: {len(mag_inst_pareada)}")

# ==========================================
# 5. CÁLCULO DO ZERO POINT
# ==========================================
# O Ponto Zero é a diferença entre a magnitude real e a instrumental
diferencas = mag_aparente_pareada - mag_inst_pareada

# Usamos a mediana para que estrelas com defeito ou variação não distorçam o resultado
zero_point = np.median(diferencas)
desvio_padrao_zp = np.std(diferencas)

print(f"Zero Point (ZP): {zero_point:.4f}")
print(f"Desvio Padrão do ZP: {desvio_padrao_zp:.4f}")

# Exemplo: aplicando o ZP para descobrir a magnitude real de uma estrela qualquer da imagem
# mag_real = mag_inst + zero_point