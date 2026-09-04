import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
from astropy.table import Table

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
fluxos_validos = tabela_fotometria['aperture_sum'] > 0
tabela_filtrada = tabela_fotometria[fluxos_validos]
fluxo = tabela_filtrada['aperture_sum']
tabela_filtrada['mag_inst'] = -2.5 * np.log10(fluxo)

# Converte o (X, Y) dessas estrelas válidas para RA e Dec
coords_imagem = wcs.pixel_to_world(tabela_filtrada['xcenter'], tabela_filtrada['ycenter'])


# ==========================================
# 3. BUSCA NO CATÁLOGO (VIZIER - UCAC4)
# ==========================================
centro_ra = wcs.pixel_to_world(image_data.shape[1]/2, image_data.shape[0]/2).ra.deg
centro_dec = wcs.pixel_to_world(image_data.shape[1]/2, image_data.shape[0]/2).dec.deg
centro_coord = SkyCoord(ra=centro_ra, dec=centro_dec, unit=(u.deg, u.deg))

# Alterado de 'Vmag' para 'Bmag'
vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'Bmag'])
vizier.ROW_LIMIT = -1 
catalogo = vizier.query_region(centro_coord, radius=15*u.arcmin, catalog='I/322A/out')[0]

# Alterado de 'Vmag' para 'Bmag'
catalogo = catalogo[~np.isnan(catalogo['Bmag'])]
coords_catalogo = SkyCoord(ra=catalogo['RAJ2000'], dec=catalogo['DEJ2000'], unit=(u.deg, u.deg))

# ==========================================
# 4. CROSS-MATCHING (CRUZAMENTO)
# ==========================================
idx_catalogo, d2d, d3d = coords_imagem.match_to_catalog_sky(coords_catalogo)
limite_distancia = 2.0 * u.arcsec
pares_validos = d2d < limite_distancia

mag_inst_pareada = tabela_filtrada['mag_inst'][pares_validos]

# Alterado de 'Vmag' para 'Bmag'
mag_aparente_pareada = catalogo['Bmag'][idx_catalogo[pares_validos]]
print(f"Pares perfeitos encontrados: {len(mag_inst_pareada)}")

# ==========================================
# 5. CÁLCULO DO ZERO POINT
# ==========================================
diferencas = mag_aparente_pareada - mag_inst_pareada
zero_point = np.median(diferencas)
desvio_padrao_zp = np.std(diferencas)

print(f"Zero Point (ZP): {zero_point:.4f}")
print(f"Desvio Padrão do ZP: {desvio_padrao_zp:.4f}")

# ==========================================
# 6. SALVAR DADOS CALIBRADOS EM CSV
# ==========================================
# Calcula a magnitude real (calibrada) para todas as estrelas da imagem
tabela_filtrada['mag_calibrada'] = tabela_filtrada['mag_inst'] + zero_point

# Cria uma tabela com RA, DEC e a Magnitude e salva
tabela_final = Table([coords_imagem.ra.deg, coords_imagem.dec.deg, tabela_filtrada['mag_calibrada']], 
                     names=('RA', 'DEC', 'Magnitude'))

# ATENÇÃO: Mude o nome ao rodar as outras bandas (fotometria_B.csv, fotometria_R.csv, etc)
nome_arquivo_saida = 'fotometria_B.csv'
tabela_final.write(nome_arquivo_saida, format='csv', overwrite=True)
print(f"Dados salvos com sucesso em: {nome_arquivo_saida}")

# ==========================================
# 7. GRÁFICO DE CALIBRAÇÃO
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(mag_inst_pareada, mag_aparente_pareada, color='royalblue', alpha=0.6, edgecolor='k', label='Estrelas Pareadas')

x_line = np.linspace(min(mag_inst_pareada), max(mag_inst_pareada), 100)
y_line = x_line + zero_point
plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label=f'Ajuste ZP ({zero_point:.2f})')

plt.xlabel('Magnitude Instrumental')
plt.ylabel('Magnitude Aparente (Catálogo VizieR)')
plt.title('Calibração Fotométrica: Instrumental vs Catálogo')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

# Único plt.show() no final do script
plt.show()