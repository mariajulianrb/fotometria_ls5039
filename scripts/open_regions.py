import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS
from regions import Regions

ARQUIVO_FITS = '/home/maju/Downloads/dados/astronometry/ls5039_G_wcs.fits'
ARQUIVO_REG = '/home/maju/Downloads/dados/scripts/fotometria/regioes_limpas_G.reg' 

dados, header = fits.getdata(ARQUIVO_FITS, header=True)
wcs = WCS(header)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1, 1, 1, projection=wcs)

norm = ImageNormalize(dados, interval=ZScaleInterval())
ax.imshow(dados, cmap='gray', origin='lower', norm=norm)

regioes = Regions.read(ARQUIVO_REG, format='ds9')
for regiao in regioes:
    regiao.plot(ax=ax)

ax.set_title('Fotometria de abertura - filtro G', fontsize=12, pad=12)

ax.coords['ra'].set_axislabel('Ascensão Reta', fontsize=11)
ax.coords['ra'].set_major_formatter('hh:mm:ss')
ax.coords['ra'].set_separator(':')
ax.coords['ra'].set_ticklabel(simplify=False)  

ax.coords['dec'].set_axislabel('Declinação', fontsize=11)
ax.coords['dec'].set_major_formatter('dd:mm:ss')
ax.coords['dec'].set_separator(('°', "'", '"'))
ax.coords['dec'].set_ticklabel(simplify=False)  

plt.tight_layout()
plt.show()