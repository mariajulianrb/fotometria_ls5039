import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats

# Configurações Iniciais
arquivo_imagem = '/home/maju/Downloads/dados/astronometry/ls5039_G_wcs.fits'
FWHM = 8.89
raio_abertura = 2.0 * FWHM
raio_in = 3.0 * FWHM
raio_out = 4.0 * FWHM
NUM_ESTRELAS_BRILHANTES = 200
MAX_ERRO_MAG = 0.05 # Filtro sugerido pelo professor para remover anéis contaminados

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
    
    # NOVO: Cálculo do Erro da Fotometria (Equação do CCD)
    area_ap = aberturas.area
    area_anel = aneis.area
    
    # Variância = Sinal da estrela + Ruído do fundo na abertura + Incerteza da medição do anel
    variancia_fluxo = fluxo_valido + (area_ap * (std_fundo ** 2)) + ((area_ap ** 2 / area_anel) * (std_fundo ** 2))
    erro_fluxo = np.sqrt(np.maximum(variancia_fluxo, 0))
    
    # Propagação do erro do fluxo para a magnitude
    erro_mag_inst = 1.0857 * (erro_fluxo / fluxo_valido)

    # 4. Salvar Tabela Intermediária com Metadados
    df_bruto = pd.DataFrame({
        'ID': fontes_validas['id'],
        'X_pix': fontes_validas['x_centroid'],
        'Y_pix': fontes_validas['y_centroid'],
        'RA_deg': coords_imagem.ra.deg,
        'Dec_deg': coords_imagem.dec.deg,
        'Fluxo': fluxo_valido,
        'Erro_Fluxo': erro_fluxo, # Coluna adicionada
        'Mag_Inst': mag_inst,
        'Erro_Mag_Inst': erro_mag_inst, # Coluna adicionada
        'Std_Fundo': std_fundo,
        'Area_Ap': area_ap,
        'Exptime': exptime
    })

    # Filtra as estrelas com erro alto (provável contaminação no anel)
    df_bruto = df_bruto[df_bruto['Erro_Mag_Inst'] <= MAX_ERRO_MAG]

    # Ordena pelo MENOR erro e mantém a quantidade desejada
    df_bruto = df_bruto.sort_values(by='Erro_Mag_Inst', ascending=True).head(NUM_ESTRELAS_BRILHANTES).reset_index(drop=True)

    df_bruto.to_csv('fotometria_bruta_G.csv', index=False)

    # Regiões DS9
    with open('regioes_aneis_semcorr_G.reg', 'w') as f:
        f.write('global color=cyan width=1 select=1 edit=1 move=1 delete=1 include=1 source=1\nimage\n')
        for x, y in zip(df_bruto['X_pix'], df_bruto['Y_pix']):
            f.write(f'circle({x+1:.2f},{y+1:.2f},{raio_abertura:.2f}) # color=cyan\n')
            f.write(f'annulus({x+1:.2f},{y+1:.2f},{raio_in:.2f},{raio_out:.2f}) # color=yellow\n')

    print(f"Sucesso! As {len(df_bruto)} estrelas com menor erro foram salvas em 'fotometria_bruta_G.csv'.")