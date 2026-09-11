import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def apply_wcs_to_image(image_filename, wcs_filename, output_filename):
    
    hdul_image = fits.open(image_filename)
    image_data = hdul_image[0].data
    image_header = hdul_image[0].header

    hdul_wcs = fits.open(wcs_filename)
    wcs_header = hdul_wcs[0].header
    
    wcs = WCS(wcs_header)

    hdul_image[0].header.update(wcs_header)

    hdul_wcs.close()

    fits.writeto(output_filename, image_data, image_header, overwrite=True)

    # Fecha o arquivo da imagem original
    hdul_image.close()

    print(f'Imagem com WCS aplicado salva como: {output_filename}')

def estimate_5sigma_limit(image_data, std):
    background_noise = std
    signal_to_noise_ratio = 10
    flux_5sigma = signal_to_noise_ratio * background_noise
    mag_5sigma = -2.5 * np.log10(flux_5sigma) + 20
    
    return mag_5sigma


imagem_sem_wcs = 'caminho/para/sua_imagem_original.fits'
wcs_baixado = 'caminho/para/seu_arquivo_baixado.wcs'
imagem_final_calibrada = 'caminho/para/sua_nova_imagem_com_wcs.fits'

apply_wcs_to_image(imagem_sem_wcs, wcs_baixado, imagem_final_calibrada)