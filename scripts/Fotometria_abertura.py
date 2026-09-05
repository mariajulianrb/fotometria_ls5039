import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# 1. Parâmetros e Leitura
FWHM_B = 7.058
with fits.open('/home/maju/Downloads/dados/astronometry/LS5039_B_wcs.fits') as hdu:
    dados = hdu[0].data.astype(float)
    header = hdu[0].header

wcs = WCS(header)
exptime = header.get('EXPTIME', 1.0)

# 2. Estatísticas do Fundo e Detecção
_, mediana, desvio = sigma_clipped_stats(dados, sigma=3.0)
daofind = DAOStarFinder(fwhm=FWHM_B, threshold=5.0 * desvio)
fontes = daofind(dados - mediana)

# 3. Remover fontes das bordas
margem = int(np.ceil(4.0 * FWHM_B))
mascara = (fontes['xcentroid'] > margem) & (fontes['xcentroid'] < dados.shape[1] - margem) & \
          (fontes['ycentroid'] > margem) & (fontes['ycentroid'] < dados.shape[0] - margem)
fontes = fontes[mascara]

# 4. Fotometria (Abertura e Anel)
posicoes = np.transpose((fontes['xcentroid'], fontes['ycentroid']))
aberturas = CircularAperture(posicoes, r=2.0 * FWHM_B)
aneis = CircularAnnulus(posicoes, r_in=3.0 * FWHM_B, r_out=4.0 * FWHM_B)

fotometria = aperture_photometry(dados, [aberturas, aneis])

# 5. Matemática do Fluxo e Magnitude
fundo_total = (fotometria['aperture_sum_1'] / aneis.area) * aberturas.area
fluxo_limpo = fotometria['aperture_sum_0'] - fundo_total

# Filtrar fontes com fluxo positivo
validos = fluxo_limpo > 0
fontes, fluxo_limpo = fontes[validos], fluxo_limpo[validos]

coords = wcs.pixel_to_world(fontes['xcentroid'], fontes['ycentroid'])
mag_inst = -2.5 * np.log10(fluxo_limpo / exptime)

# 6. Salvar Tabela CSV
df = pd.DataFrame({
    'ID': fontes['id'], 'X_pix': fontes['xcentroid'], 'Y_pix': fontes['ycentroid'],
    'RA_deg': coords.ra.deg, 'Dec_deg': coords.dec.deg, 
    'Fluxo': fluxo_limpo, 'Mag_Inst': mag_inst
})
df.to_csv('resultado_fotometria.csv', index=False)
print(f"{len(df)} estrelas fotometradas e salvas em CSV.")

# 7. Salvar Regiões para o DS9
with open('centroides.reg', 'w') as f:
    f.write('global color=cyan width=1 select=1 edit=1 move=1 delete=1 include=1 source=1\nimage\n')
    for x, y in zip(fontes['xcentroid'], fontes['ycentroid']):
        x_d, y_d = x + 1.0, y + 1.0
        f.write(f'circle({x_d:.2f},{y_d:.2f},{2.0*FWHM_B:.2f}) # color=cyan\n')
        f.write(f'circle({x_d:.2f},{y_d:.2f},{3.0*FWHM_B:.2f}) # color=green\n')
        f.write(f'circle({x_d:.2f},{y_d:.2f},{4.0*FWHM_B:.2f}) # color=green\n')
        f.write(f'point({x_d:.2f},{y_d:.2f}) # point=cross color=red\n')