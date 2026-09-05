import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip, sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astroquery.vizier import Vizier
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# ==========================================
# CONFIGURAÇÕES INICIAIS
# ==========================================
arquivo_imagem = '/home/maju/Downloads/dados/astronometry/LS5039_B_wcs.fits'
FWHM = 5.0
raio_abertura = FWHM             # Raio para medir a luz da estrela
raio_in = FWHM * 1.5             # Início do anel de medição do céu
raio_out = FWHM * 2.0            # Fim do anel de medição do céu

with fits.open(arquivo_imagem) as hdul:
    image_data = hdul[0].data.astype(float)
    header = hdul[0].header

wcs = WCS(header)
exptime = header.get('EXPTIME', 1.0)

# ==========================================
# 1. RUÍDO DE FUNDO 2D E DETECÇÃO ROBUSTA
# ==========================================
# O mapa 2D resolve problemas de vinhetagem e iluminação irregular
bkg = Background2D(
    image_data, box_size=(64, 64), filter_size=(3, 3),
    sigma_clip=SigmaClip(sigma=3.0), bkg_estimator=MedianBackground()
)
dados_subtraidos = image_data - bkg.background

# Detecção com filtros geométricos para evitar falsos positivos
daofind = DAOStarFinder(
    fwhm=FWHM, 
    threshold=5.0 * bkg.background_rms_median,
    sharplo=0.2, sharphi=1.0, 
    roundlo=-0.5, roundhi=0.5
)
fontes = daofind(dados_subtraidos)
print(f"Estrelas detectadas (validadas): {len(fontes)}")

# ==========================================
# 2. FOTOMETRIA DE ABERTURA COM ANEL DE CÉU
# ==========================================
posicoes = np.transpose((fontes['xcentroid'], fontes['ycentroid']))
aberturas = CircularAperture(posicoes, r=raio_abertura)
aneis = CircularAnnulus(posicoes, r_in=raio_in, r_out=raio_out)

tabela_fotometria = aperture_photometry(image_data, [aberturas, aneis])

# Subtração do fundo local medido no anel de cada estrela
fundo_medio_anel = tabela_fotometria['aperture_sum_1'] / aneis.area
fundo_total_abertura = fundo_medio_anel * aberturas.area
fluxo_limpo = tabela_fotometria['aperture_sum_0'] - fundo_total_abertura

# Filtrar fluxos negativos ou nulos
validos = fluxo_limpo > 0
fluxo_valido = fluxo_limpo[validos]
fontes_validas = fontes[validos]

# Magnitude instrumental e coordenadas
mag_inst = -2.5 * np.log10(fluxo_valido / exptime)
coords_imagem = wcs.pixel_to_world(fontes_validas['xcentroid'], fontes_validas['ycentroid'])

# ==========================================
# 3. BUSCA NO CATÁLOGO (VIZIER - UCAC4)
# ==========================================
centro_ra = wcs.pixel_to_world(image_data.shape[1]/2, image_data.shape[0]/2).ra.deg
centro_dec = wcs.pixel_to_world(image_data.shape[1]/2, image_data.shape[0]/2).dec.deg
centro_coord = SkyCoord(ra=centro_ra, dec=centro_dec, unit=(u.deg, u.deg))

print("\nBaixando referências do catálogo UCAC4...")
vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'Bmag'], row_limit=-1)
catalogo = vizier.query_region(centro_coord, radius=15*u.arcmin, catalog='I/322A/out')[0]

catalogo = catalogo[~np.isnan(catalogo['Bmag'])]
coords_catalogo = SkyCoord(ra=catalogo['RAJ2000'], dec=catalogo['DEJ2000'], unit=(u.deg, u.deg))

# ==========================================
# 4. CROSS-MATCHING ESPACIAL
# ==========================================
idx_catalogo, d2d, _ = coords_imagem.match_to_catalog_sky(coords_catalogo)
pares_validos = d2d < (2.0 * u.arcsec)

mag_inst_pareada = mag_inst[pares_validos]
mag_aparente_pareada = catalogo['Bmag'][idx_catalogo[pares_validos]]
print(f"Pares perfeitos cruzados: {len(mag_inst_pareada)}")

# ==========================================
# 5. CÁLCULO DO ZERO POINT (COM SIGMA CLIPPING)
# ==========================================
diferencas = mag_aparente_pareada - mag_inst_pareada

# O sigma clipping remove estrelas variáveis ou cruzas erradas que poluem o ZP
diferencas_limpas = sigma_clip(diferencas, sigma=2.5)

zero_point = np.ma.median(diferencas_limpas)
desvio_padrao_zp = np.ma.std(diferencas_limpas)

print(f"Zero Point (ZP): {zero_point:.4f}")
print(f"Desvio Padrão do ZP: {desvio_padrao_zp:.4f}\n")

# ==========================================
# 6. SALVAR DADOS CALIBRADOS EM CSV
# ==========================================
mag_calibrada_total = mag_inst + zero_point

tabela_final = Table([
    coords_imagem.ra.deg, 
    coords_imagem.dec.deg, 
    fluxo_valido, 
    mag_inst, 
    mag_calibrada_total
], names=('RA', 'DEC', 'Fluxo', 'Mag_Inst', 'Mag_B_Calibrada'))

nome_arquivo_saida = 'fotometria_B_calibrada.csv'
tabela_final.write(nome_arquivo_saida, format='csv', overwrite=True)
print(f"Resultados de {len(tabela_final)} estrelas salvos em: {nome_arquivo_saida}")

# ==========================================
# 7. GRÁFICO DE CALIBRAÇÃO (INSTRUMENTAL x REAL)
# ==========================================
# Separar inliers e outliers para visualização no gráfico
mascara_inliers = ~diferencas_limpas.mask
mascara_outliers = diferencas_limpas.mask

plt.figure(figsize=(9, 6))

# Plotar pontos aceitos
plt.scatter(mag_inst_pareada[mascara_inliers], mag_aparente_pareada[mascara_inliers], 
            color='royalblue', alpha=0.7, edgecolor='k', label='Estrelas Válidas (Inliers)')

# Plotar pontos rejeitados (para controle de qualidade)
if mascara_outliers.any():
    plt.scatter(mag_inst_pareada[mascara_outliers], mag_aparente_pareada[mascara_outliers], 
                color='red', alpha=0.7, marker='x', label='Descartadas (Outliers)')

# Linha de tendência baseada no ZP
x_line = np.linspace(min(mag_inst_pareada), max(mag_inst_pareada), 100)
y_line = x_line + zero_point
plt.plot(x_line, y_line, color='darkorange', linestyle='--', linewidth=2.5, 
         label=f'Ajuste ZP = {zero_point:.2f} $\\pm$ {desvio_padrao_zp:.2f}')

plt.xlabel('Magnitude Instrumental')
plt.ylabel('Magnitude Aparente ($B_{mag}$ Catálogo UCAC4)')
plt.title('Calibração Fotométrica: Instrumental vs Catálogo')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()