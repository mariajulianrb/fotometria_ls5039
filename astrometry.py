import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# 1. Nome do arquivo FITS da banda individual
arquivo_fits = 'ls5039_R_astrometry.fits'

# 2. Carregar dados e cabeçalho FITS
with fits.open(arquivo_fits) as hdu:
    dados = hdu[0].data.astype(float)
    header = hdu[0].header

# Obter tempo de exposição e WCS
exptime = header.get('EXPTIME', 1.0)
wcs = WCS(header)

# 3. Estatísticas do fundo do céu
media, mediana, desvio = sigma_clipped_stats(dados, sigma=3.0)

# 4. Detectar estrelas na imagem
daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * desvio)
fontes = daofind(dados - mediana)

# 5. Definir abertura da estrela (raio 5px) e anel de fundo (10 a 15px)
posicoes = np.transpose((fontes['xcentroid'], fontes['ycentroid']))
aberturas = CircularAperture(posicoes, r=5.0)
aneis = CircularAnnulus(posicoes, r_in=10.0, r_out=15.0)

# 6. Calcular a fotometria de abertura e subtrair o fundo local
tabela_fot = aperture_photometry(dados, [aberturas, aneis])
fundo_medio = tabela_fot['aperture_sum_1'] / aneis.area
fundo_total = fundo_medio * aberturas.area
fluxo_limpo = tabela_fot['aperture_sum_0'] - fundo_total

# 7. Filtrar apenas fluxos positivos e calcular a Magnitude Instrumental
mascara = fluxo_limpo > 0
fontes_validas = fontes[mascara]
fluxo_valido = fluxo_limpo[mascara]

# Fórmula: -2.5 * log10(fluxo / exptime) + 25
mag_inst = -2.5 * np.log10(fluxo_valido / exptime) + 25.0

# 8. Converter pixels (X, Y) para RA e Dec
coords = wcs.pixel_to_world(fontes_validas['xcentroid'], fontes_validas['ycentroid'])

# 9. Gerar a tabela final em DataFrame
df = pd.DataFrame({
    'ID': fontes_validas['id'],
    'X_pix': fontes_validas['xcentroid'],
    'Y_pix': fontes_validas['ycentroid'],
    'RA_deg': coords.ra.deg,
    'Dec_deg': coords.dec.deg,
    'Fluxo': fluxo_valido,
    'Mag_Inst': mag_inst
})

# Salvar o arquivo CSV no computador
nome_csv = 'resultado_fotometria_R.csv'
df.to_csv(nome_csv, index=False)

# 10. Salvar arquivo de regiões para o DS9
nome_reg = 'centroides_R.reg'
with open(nome_reg, 'w') as f:
    f.write('# Region file format: DS9 version 4.1\n')
    f.write('global color=cyan width=1 select=1 edit=1 move=1 delete=1 include=1 source=1\n')
    f.write('image\n')
    
    for x, y in zip(fontes_validas['xcentroid'], fontes_validas['ycentroid']):
        # Conversão de índice 0 (Python) para índice 1 (DS9)
        x_ds9 = x + 1.0
        y_ds9 = y + 1.0
        # Desenha a abertura fotométrica (círculo) e o centroide (cruz vermelha)
        f.write(f'circle({x_ds9:.2f},{y_ds9:.2f},5.0) # color=cyan\n')
        f.write(f'point({x_ds9:.2f},{y_ds9:.2f}) # point=cross color=red\n')

# Exibir resumo no terminal
print(f"Processamento concluído! {len(df)} estrelas detectadas.")
print(f"Tabela salva em: {nome_csv}")
print(f"Arquivo de regiões para DS9 salvo em: {nome_reg}")
print("\nPrimeiras linhas da tabela gerada:")
print(df.head())