import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# 1. Função para aplicar o WCS que você baixou na sua imagem original
def apply_wcs_to_image(image_filename, wcs_filename, output_filename):
    # Abre o arquivo FITS da imagem original
    hdul_image = fits.open(image_filename)
    image_data = hdul_image[0].data
    image_header = hdul_image[0].header

    # Abre o arquivo WCS baixado do site astrometry.net
    hdul_wcs = fits.open(wcs_filename)
    wcs_header = hdul_wcs[0].header
    
    # Valida e prepara o WCS
    wcs = WCS(wcs_header)

    # Atualiza o header da imagem original com as informações do WCS
    hdul_image[0].header.update(wcs_header)

    # Fecha o arquivo WCS
    hdul_wcs.close()

    # Salva a imagem com o novo nome e WCS aplicado
    fits.writeto(output_filename, image_data, image_header, overwrite=True)

    # Fecha o arquivo da imagem original
    hdul_image.close()

    print(f'Imagem com WCS aplicado salva como: {output_filename}')

# 2. Função para estimar a magnitude limite do instrumento (5-sigma)
def estimate_5sigma_limit(image_data, std):
    # Calcula o limite de detecção para 5-sigma
    background_noise = std
    signal_to_noise_ratio = 10
    flux_5sigma = signal_to_noise_ratio * background_noise
    mag_5sigma = -2.5 * np.log10(flux_5sigma) + 20
    
    return mag_5sigma


# --- COMO EXECUTAR SEUS PRÓXIMOS PASSOS ---

# Substitua os caminhos abaixo pelos arquivos reais que você tem no seu computador
imagem_sem_wcs = 'caminho/para/sua_imagem_original.fits'
wcs_baixado = 'caminho/para/seu_arquivo_baixado.wcs'
imagem_final_calibrada = 'caminho/para/sua_nova_imagem_com_wcs.fits'

# Execute a função para unir os dois arquivos
apply_wcs_to_image(imagem_sem_wcs, wcs_baixado, imagem_final_calibrada)