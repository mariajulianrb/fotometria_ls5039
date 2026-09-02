import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# Configuração da banda atual ('B', 'G' ou 'R')
banda = 'B'

arquivo_fits = f'ls5039_{banda.lower()}_wcs_final.fits'
nome_csv = f'resultado_fotometria_{banda}.csv'
nome_reg = f'centroides_{banda}.reg'

# 1. Carregar dados e cabeçalho FITS
with fits.open(arquivo_fits) as hdu:
    dados = hdu[0].data.astype(float)
    header = hdu[0].header

exptime = header.get('EXPTIME', 1.0)
wcs = WCS(header)

# 2. Estatísticas do fundo do céu
media, mediana, desvio = sigma_clipped_stats(dados, sigma=3.0)

# 3. Detectar estrelas na imagem
daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * desvio)
fontes = daofind(dados - mediana)

# 4. Descartar borda vinhetada (Margem ampliada para 220px)
margem = 220
altura, largura = dados.shape
mascara_borda = (
    (fontes['xcentroid'] > margem) & 
    (fontes['xcentroid'] < largura - margem) & 
    (fontes['ycentroid'] > margem) & 
    (fontes['ycentroid'] < altura - margem)
)
fontes = fontes[mascara_borda]

# 5. Definir abertura da estrela (r = 5px) e anel de fundo local (r_in = 10px, r_out = 15px)
posicoes = np.transpose((fontes['xcentroid'], fontes['ycentroid']))
aberturas = CircularAperture(posicoes, r=5.0)
aneis = CircularAnnulus(posicoes, r_in=10.0, r_out=15.0)

# 6. Fotometria de abertura e subtração do fundo local
tabela_fot = aperture_photometry(dados, [aberturas, aneis])
fundo_medio = tabela_fot['aperture_sum_1'] / aneis.area
fundo_total = fundo_medio * aberturas.area
fluxo_limpo = tabela_fot['aperture_sum_0'] - fundo_total

# 7. Filtrar fluxos válidos e calcular a Magnitude Instrumental
mascara_fluxo = fluxo_limpo > 0
fontes_validas = fontes[mascara_fluxo]
fluxo_valido = fluxo_limpo[mascara_fluxo]

mag_inst = -2.5 * np.log10(fluxo_valido / exptime) + 25.0

# 8. Converter pixels (X, Y) para RA e Dec
coords = wcs.pixel_to_world(fontes_validas['xcentroid'], fontes_validas['ycentroid'])

# 9. Gerar a tabela de saída
df = pd.DataFrame({
    'ID': fontes_validas['id'],
    'X_pix': fontes_validas['xcentroid'],
    'Y_pix': fontes_validas['ycentroid'],
    'RA_deg': coords.ra.deg,
    'Dec_deg': coords.dec.deg,
    'Fluxo': fluxo_valido,
    'Mag_Inst': mag_inst
})

df.to_csv(nome_csv, index=False)

# 10. Salvar arquivo de regiões do DS9
with open(nome_reg, 'w') as f:
    f.write('# Region file format: DS9 version 4.1\n')
    f.write('global color=cyan width=1 select=1 edit=1 move=1 delete=1 include=1 source=1\n')
    f.write('image\n')
    
    for x, y in zip(fontes_validas['xcentroid'], fontes_validas['ycentroid']):
        x_ds9 = x + 1.0
        y_ds9 = y + 1.0
        f.write(f'circle({x_ds9:.2f},{y_ds9:.2f},5.0) # color=cyan\n')
        f.write(f'point({x_ds9:.2f},{y_ds9:.2f}) # point=cross color=red\n')

print(f"Processamento da Banda {banda} concluído!")
print(f"Total: {len(df)} estrelas salvas (borda de {margem}px removida).")