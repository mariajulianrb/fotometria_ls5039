import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# 1. Configurações Iniciais e Leitura do Arquivo FITS
FWHM_B = 7.058
arquivo = '/home/maju/Downloads/dados/astronometry/LS5039_B_wcs.fits'

with fits.open(arquivo) as hdu:
    dados = hdu[0].data.astype(float)
    header = hdu[0].header

wcs = WCS(header)
exptime = header.get('EXPTIME', 1.0)

# 2. Mapeamento 2D do Fundo do Céu (Remove Gradientes e Vinhetagem)
bkg = Background2D(
    dados, 
    box_size=(64, 64), 
    filter_size=(3, 3),
    sigma_clip=SigmaClip(sigma=3.0), 
    bkg_estimator=MedianBackground()
)
dados_sub = dados - bkg.background

# 3. Detecção com Filtros de Nitidez e Geometria Estelar
daofind = DAOStarFinder(
    fwhm=FWHM_B,
    threshold=100.0,            # Limiar absoluto de brilho acima do fundo (ADU)
    sharplo=0.3, sharphi=0.9,   # Descarta ruídos extremamente pontuais
    roundlo=-0.5, roundhi=0.5   # Exige simetria circular (descarta riscos)
)
fontes = daofind(dados_sub)

if fontes is None:
    print("Nenhuma fonte encontrada. Considere reduzir o 'threshold'.")
else:
    # 4. Corte Dinâmico de Bordas
    margem = int(np.ceil(4.0 * FWHM_B))
    altura, largura = dados.shape
    mascara_borda = (
        (fontes['xcentroid'] > margem) & (fontes['xcentroid'] < largura - margem) &
        (fontes['ycentroid'] > margem) & (fontes['ycentroid'] < altura - margem)
    )
    fontes = fontes[mascara_borda]

    # 5. Fotometria de Abertura e Subtração do Fundo do Céu
    posicoes = np.transpose((fontes['xcentroid'], fontes['ycentroid']))
    aberturas = CircularAperture(posicoes, r=2.0 * FWHM_B)
    aneis = CircularAnnulus(posicoes, r_in=3.0 * FWHM_B, r_out=4.0 * FWHM_B)

    fotometria = aperture_photometry(dados, [aberturas, aneis])

    fundo_total = (fotometria['aperture_sum_1'] / aneis.area) * aberturas.area
    fluxo_limpo = fotometria['aperture_sum_0'] - fundo_total

    # Filtra apenas detecções com sinal significativo
    validos = fluxo_limpo > 50.0
    fontes, fluxo_limpo = fontes[validos], fluxo_limpo[validos]

    # 6. Coordenadas WCS e Magnitudes Instrumentais (Pogson)
    coords = wcs.pixel_to_world(fontes['xcentroid'], fontes['ycentroid'])
    mag_inst = -2.5 * np.log10(fluxo_limpo / exptime)

    # 7. Montagem do Dataframe e Filtro das Top 100 Mais Brilhantes
    df = pd.DataFrame({
        'ID': fontes['id'],
        'X_pix': fontes['xcentroid'],
        'Y_pix': fontes['ycentroid'],
        'RA_deg': coords.ra.deg,
        'Dec_deg': coords.dec.deg,
        'Fluxo': fluxo_limpo,
        'Mag_Inst': mag_inst
    })

    # Ordena pelo fluxo decrescente e retém as 400 fontes com maior brilho
    df = df.sort_values(by='Fluxo', ascending=False).head(400).reset_index(drop=True)

    # 8. Exportação (CSV e Arquivo de Regiões do DS9)
    df.to_csv('resultado_fotometria_B.csv', index=False)

    with open('centroides_B.reg', 'w') as f:
        f.write('global color=cyan width=1 select=1 edit=1 move=1 delete=1 include=1 source=1\nimage\n')
        for x, y in zip(df['X_pix'], df['Y_pix']):
            f.write(f'circle({x+1:.2f},{y+1:.2f},{2.0*FWHM_B:.2f}) # color=cyan\n')
            f.write(f'point({x+1:.2f},{y+1:.2f}) # point=cross color=red\n')

    print(f"Sucesso! As {len(df)} estrelas mais brilhantes foram salvas em CSV e no arquivo .reg do DS9.")