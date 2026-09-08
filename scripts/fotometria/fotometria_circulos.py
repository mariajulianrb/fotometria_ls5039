import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# 1. Configurações Iniciais e Leitura
FWHM_B = 7.058
arquivo = '/home/maju/Downloads/dados/astronometry/LS5039_B_wcs.fits'

with fits.open(arquivo) as hdu:
    dados = hdu[0].data.astype(float)
    header = hdu[0].header

wcs = WCS(header)
exptime = header.get('EXPTIME', 1.0)

# 2. Estatística e Detecção
_, mediana, desvio = sigma_clipped_stats(dados, sigma=3.0)

daofind = DAOStarFinder(
    fwhm=FWHM_B, 
    threshold=10.0 * desvio,
    sharplo=0.2, sharphi=1.0,  # Ignora picos irreais ou borrões gigantes
    roundlo=-1.0, roundhi=1.0  # Ignora riscos ou manchas muito alongadas
)
fontes = daofind(dados - mediana)

# 3. Filtragem de Bordas (Margem Fixa = 25)
altura, largura = dados.shape
mascara = (fontes['xcentroid'] > 25) & (fontes['xcentroid'] < largura - 25) & \
          (fontes['ycentroid'] > 25) & (fontes['ycentroid'] < altura - 25)
fontes = fontes[mascara]

# 4. Fotometria (Abertura e Anel do Fundo)
posicoes = np.transpose((fontes['xcentroid'], fontes['ycentroid']))
aberturas = CircularAperture(posicoes, r=2.0 * FWHM_B)
aneis = CircularAnnulus(posicoes, r_in=3.0 * FWHM_B, r_out=4.0 * FWHM_B)

fotometria = aperture_photometry(dados, [aberturas, aneis])

# 5. Cálculo do Fluxo Líquido e Magnitude Instrumental
fundo_total = (fotometria['aperture_sum_1'] / aneis.area) * aberturas.area
fluxo_limpo = fotometria['aperture_sum_0'] - fundo_total

# Manter apenas fontes com fluxo positivo
validos = fluxo_limpo > (10 * desvio)
fontes, fluxo_limpo = fontes[validos], fluxo_limpo[validos]


# Conversão de Coordenadas e Pogson
coords = wcs.pixel_to_world(fontes['xcentroid'], fontes['ycentroid'])
mag_inst = -2.5 * np.log10(fluxo_limpo / exptime)

# 6. Exportação (CSV e DS9)
df = pd.DataFrame({
    'ID': fontes['id'], 'X_pix': fontes['xcentroid'], 'Y_pix': fontes['ycentroid'],
    'RA_deg': coords.ra.deg, 'Dec_deg': coords.dec.deg, 
    'Fluxo': fluxo_limpo, 'Mag_Inst': mag_inst
})
df.to_csv('resultado_fotometria_B.csv', index=False)

with open('centroides_B.reg', 'w') as f:
    f.write('global color=cyan width=1 select=1 edit=1 move=1 delete=1 include=1 source=1\nimage\n')
    for x, y in zip(fontes['xcentroid'], fontes['ycentroid']):
        f.write(f'circle({x+1:.2f},{y+1:.2f},{2.0*FWHM_B:.2f}) # color=cyan\n')
        f.write(f'point({x+1:.2f},{y+1:.2f}) # point=cross color=red\n')
        
print(f"Fotometria concluída com sucesso: {len(df)} estrelas salvas no CSV e arquivo de região gerado.")