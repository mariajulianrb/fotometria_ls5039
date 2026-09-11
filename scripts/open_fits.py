import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS

#ARQUIVO_FITS = '/home/maju/Downloads/dados/astronometry/ls5039_B_wcs.fits'
#ARQUIVO_FITS = '/home/maju/Downloads/dados/astronometry/ls5039_R_wcs.fits'
ARQUIVO_FITS = '/home/maju/Downloads/dados/astronometry/ls5039_G_wcs.fits'

dados, header = fits.getdata(ARQUIVO_FITS, header=True)
wcs = WCS(header)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1, 1, 1, projection=wcs)

norm = ImageNormalize(dados, interval=ZScaleInterval())
ax.imshow(dados, cmap='gray', origin='lower', norm=norm)

ax.set_title('Solução astrométrica para o filtro G', fontsize=12, pad=12)

ax.coords['ra'].set_axislabel('Ascensão Reta (J2000)', fontsize=11)
ax.coords['ra'].set_major_formatter('hh:mm:ss')
ax.coords['ra'].set_separator(':')
ax.coords['ra'].set_ticklabel(simplify=False)  

ax.coords['dec'].set_axislabel('Declinação (J2000)', fontsize=11)
ax.coords['dec'].set_major_formatter('dd:mm:ss')
ax.coords['dec'].set_separator(('°', "'", '"'))
ax.coords['dec'].set_ticklabel(simplify=False)  
plt.tight_layout()
plt.show()