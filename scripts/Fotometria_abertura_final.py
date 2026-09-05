import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats

# Configurações Iniciais
arquivo_imagem = '/home/maju/Downloads/dados/astronometry/LS5039_B_wcs.fits'
FWHM = 7.058
raio_abertura = 2.0 * FWHM
raio_in = 3.0 * FWHM
raio_out = 4.0 * FWHM

# 1. Carregamento da Imagem e WCS
with fits.open(arquivo_imagem) as hdul:
    image_data = hdul[0].data.astype(float)
    header = hdul[0].header

wcs = WCS(header)
exptime = header.get('EXPTIME', 1.0)
altura, largura = image_data.shape

# 2. Tratamento do Fundo e Detecção
bkg = Background2D(
    image_data, box_size=(64, 64), filter_size=(3, 3),
    sigma_clip=SigmaClip(sigma=3.0), bkg_estimator=MedianBackground()
)
dados_subtraidos = image_data - bkg.background
_, _, std_fundo = sigma_clipped_stats(dados_subtraidos, sigma=3.0)

daofind = DAOStarFinder(
    fwhm=FWHM, 
    threshold=10.0 * std_fundo,
    sharpness_range=(0.3, 1.0),
    roundness_range=(-0.5, 0.5)
)
fontes = daofind(dados_subtraidos)

if fontes is None:
    print("Nenhuma fonte encontrada.")
else:
    # Corte de borda
    margem = int(np.ceil(raio_out))
    mascara_borda = (
        (fontes['x_centroid'] > margem) & (fontes['x_centroid'] < largura - margem) &
        (fontes['y_centroid'] > margem) & (fontes['y_centroid'] < altura - margem)
    )
    fontes = fontes[mascara_borda]

    # 3. Fotometria de Abertura com Anel (Mediana)
    posicoes = np.transpose((fontes['x_centroid'], fontes['y_centroid']))
    aberturas = CircularAperture(posicoes, r=raio_abertura)
    aneis = CircularAnnulus(posicoes, r_in=raio_in, r_out=raio_out)

    tabela_fotometria = aperture_photometry(image_data, aberturas)
    estatisticas_anel = ApertureStats(image_data, aneis, sigma_clip=SigmaClip(sigma=3.0))

    fundo_total_abertura = estatisticas_anel.median * aberturas.area
    fluxo_limpo = tabela_fotometria['aperture_sum'] - fundo_total_abertura

    validos = fluxo_limpo > 0
    fluxo_valido = fluxo_limpo[validos]
    fontes_validas = fontes[validos]

    mag_inst = -2.5 * np.log10(fluxo_valido / exptime)
    coords_imagem = wcs.pixel_to_world(fontes_validas['x_centroid'], fontes_validas['y_centroid'])

    # 4. Salvar Tabela Intermediária com Metadados
    df_bruto = pd.DataFrame({
        'ID': fontes_validas['id'],
        'X_pix': fontes_validas['x_centroid'],
        'Y_pix': fontes_validas['y_centroid'],
        'RA_deg': coords_imagem.ra.deg,
        'Dec_deg': coords_imagem.dec.deg,
        'Fluxo': fluxo_valido,
        'Mag_Inst': mag_inst,
        'Std_Fundo': std_fundo,
        'Area_Ap': aberturas.area,
        'Exptime': exptime
    }).sort_values(by='Fluxo', ascending=False).reset_index(drop=True)

    df_bruto.to_csv('fotometria_bruta_B.csv', index=False)

    # Regiões DS9
    with open('regioes_aneis_B.reg', 'w') as f:
        f.write('global color=cyan width=1 select=1 edit=1 move=1 delete=1 include=1 source=1\nimage\n')
        for x, y in zip(df_bruto['X_pix'], df_bruto['Y_pix']):
            f.write(f'circle({x+1:.2f},{y+1:.2f},{raio_abertura:.2f}) # color=cyan\n')
            f.write(f'annulus({x+1:.2f},{y+1:.2f},{raio_in:.2f},{raio_out:.2f}) # color=yellow\n')

    print(f"Sucesso! {len(df_bruto)} fontes processadas e salvas em 'fotometria_bruta_B.csv'.")