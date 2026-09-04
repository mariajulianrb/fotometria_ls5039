import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# Configuração da banda atual ('B', 'G' ou 'R')
banda = 'B'

arquivo_fits = f'/home/maju/Downloads/dados/astronometry/LS5039_{banda}_wcs.fits'
nome_csv = f'resultado_fotometria_{banda}.csv'
nome_reg = f'centroides_{banda}.reg'

# 1. Carregar dados e cabeçalho FITS

with fits.open(arquivo_fits) as hdu:
    dados = hdu[0].data.astype(float)
    header = hdu[0].header.copy()  # O .copy() garante que tudo fique na memória após fechar o arquivo

# Busca e imprime as chaves mais comuns de seeing/FWHM
chaves_fwhm = ['SEEING', 'FWHM', 'PSF_FWHM', 'DIMMSEE']
for chave in chaves_fwhm:
    if chave in header:
        print(f"Encontrado {chave}: {header[chave]}")