import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from scipy.spatial import distance

# Configurações Iniciais
arquivo_imagem = '/home/maju/Downloads/dados/astronometry/ls5039_B_wcs.fits'
FWHM = 8.89
raio_abertura = 2.0 * FWHM
raio_in = 3.0 * FWHM
raio_out = 4.0 * FWHM
NUM_ESTRELAS_BRILHANTES = 250
MIN_SNR = 20.0  # SNR >= 20 equivale a um erro <= 0.05 mag

# 1. Carregamento da Imagem e WCS
with fits.open(arquivo_imagem) as hdul:
    image_data = hdul[0].data.astype(float)
    header = hdul[0].header

wcs = WCS(header)
exptime = header.get('EXPTIME', 1.0)
altura, largura = image_data.shape

# 2. Tratamento do Fundo e Detecção
bkg = Background2D(
    image_data,
    box_size=(64, 64),
    filter_size=(3, 3),
    sigma_clip=SigmaClip(sigma=3.0),
    bkg_estimator=MedianBackground(),
)
dados_subtraidos = image_data - bkg.background
_, _, std_fundo_global = sigma_clipped_stats(dados_subtraidos, sigma=3.0)

daofind = DAOStarFinder(
    fwhm=FWHM,
    threshold=10.0 * std_fundo_global,
    sharpness_range=(0.3, 1.0),
    roundness_range=(-0.5, 0.5),
)
fontes = daofind(dados_subtraidos)

if fontes is None:
    print('Nenhuma fonte encontrada.')
else:
    # Corte de borda
    margem = int(np.ceil(raio_out))
    mascara_borda = (
        (fontes['x_centroid'] > margem)
        & (fontes['x_centroid'] < largura - margem)
        & (fontes['y_centroid'] > margem)
        & (fontes['y_centroid'] < altura - margem)
    )
    fontes = fontes[mascara_borda]

    # Filtro de isolamento espacial
    coords_pixel = np.transpose((fontes['x_centroid'], fontes['y_centroid']))
    matriz_dist = distance.cdist(coords_pixel, coords_pixel)
    np.fill_diagonal(matriz_dist, np.inf)

    dist_minima = np.min(matriz_dist, axis=1)
    mascara_isoladas = dist_minima >= raio_out
    fontes = fontes[mascara_isoladas]

    # 3. Fotometria de Abertura
    posicoes = np.transpose((fontes['x_centroid'], fontes['y_centroid']))
    aberturas = CircularAperture(posicoes, r=raio_abertura)
    aneis = CircularAnnulus(posicoes, r_in=raio_in, r_out=raio_out)

    tabela_fotometria = aperture_photometry(image_data, aberturas)
    estatisticas_anel = ApertureStats(
        image_data, aneis, sigma_clip=SigmaClip(sigma=3.0)
    )

    fundo_total_abertura = estatisticas_anel.median * aberturas.area
    fluxo_limpo = tabela_fotometria['aperture_sum'] - fundo_total_abertura

    validos = fluxo_limpo > 0
    fluxo_valido = fluxo_limpo[validos]
    fontes_validas = fontes[validos]

    # --- CÁLCULO DIRETO E PADRÃO DE SNR E ERRO ---
    std_fundo_local = estatisticas_anel.std[validos]
    area_ap = aberturas.area

    # Ruído do CCD clássico: Poisson da fonte + Ruído de fundo na abertura
    erro_fluxo = np.sqrt(fluxo_valido + area_ap * (std_fundo_local**2))

    # SNR e Erro em Magnitude
    snr = fluxo_valido / erro_fluxo
    mag_inst = -2.5 * np.log10(fluxo_valido / exptime)
    erro_mag_inst = 1.0857 / snr

    coords_imagem = wcs.pixel_to_world(
        fontes_validas['x_centroid'], fontes_validas['y_centroid']
    )

    df_bruto = pd.DataFrame({
        'ID': fontes_validas['id'],
        'X_pix': fontes_validas['x_centroid'],
        'Y_pix': fontes_validas['y_centroid'],
        'RA_deg': coords_imagem.ra.deg,
        'Dec_deg': coords_imagem.dec.deg,
        'Fluxo': fluxo_valido,
        'Erro_Fluxo': erro_fluxo,
        'SNR': snr,
        'Mag_Inst': mag_inst,
        'Erro_Mag_Inst': erro_mag_inst,
    })

    # Filtro direto por SNR
    df_bruto = df_bruto[df_bruto['SNR'] >= MIN_SNR]

    # Ordena pelas mais brilhantes
    df_bruto = (
        df_bruto.sort_values(by='Fluxo', ascending=False)
        .head(NUM_ESTRELAS_BRILHANTES)
        .reset_index(drop=True)
    )

    df_bruto.to_csv('fotometria_final_B.csv', index=False)

    # Regiões DS9
    # Gerar arquivo .reg com o ID da tabela impresso na imagem
    with open('regioes_aneis_B.reg', 'w') as f:
        f.write(
            'global color=cyan width=1 select=1 edit=1 move=1 delete=1 include=1'
            ' source=1\nimage\n'
        )
        for id_est, x, y in zip(
            df_bruto['ID'], df_bruto['X_pix'], df_bruto['Y_pix']
        ):
            # Escreve o círculo e coloca o número do ID da tabela ao lado
            f.write(
                f'circle({x+1:.2f},{y+1:.2f},{raio_abertura:.2f}) # color=cyan'
                f' text={{{id_est}}}\n'
            )
            f.write(
                f'annulus({x+1:.2f},{y+1:.2f},{raio_in:.2f},{raio_out:.2f}) #'
                ' color=yellow\n'
            )

    print(
        f'Sucesso! {len(df_bruto)} fontes com SNR >= {MIN_SNR} foram salvas.'
    )