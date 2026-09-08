import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from scipy.spatial import distance

ARQUIVO = '/home/maju/Downloads/dados/astronometry/ls5039_B_wcs.fits'
FWHM = 8.89
RAIO_AP = 1.5 * FWHM
RAIO_IN, RAIO_OUT = 3.0 * FWHM, 4.0 * FWHM
MIN_SNR = 10.0 

dados, header = fits.getdata(ARQUIVO, header=True)
dados = dados.astype(float)
wcs = WCS(header)
exptime = header.get('EXPTIME', 1.0)
altura, largura = dados.shape

bkg = Background2D(
    dados, box_size=(64, 64), filter_size=(3, 3),
    sigma_clip=SigmaClip(3.0), bkg_estimator=MedianBackground()
)
dados_sub = dados - bkg.background
_, _, std_fundo = sigma_clipped_stats(dados_sub, sigma=3.0)


daofind = DAOStarFinder(
    fwhm=FWHM,
    threshold=10.0 * std_fundo, 
    sharpness_range=(0.2, 2.0), 
    roundness_range=(-1.0, 1.0)
)
fontes = daofind(dados_sub)
posicoes = np.transpose((fontes['x_centroid'], fontes['y_centroid']))


aberturas = CircularAperture(posicoes, r=RAIO_AP)
aneis = CircularAnnulus(posicoes, r_in=RAIO_IN, r_out=RAIO_OUT)

fotometria = aperture_photometry(dados, aberturas)
estat_anel = ApertureStats(dados, aneis, sigma_clip=SigmaClip(3.0))

# Cálculos fotométricos base
fundo_total = estat_anel.median * aberturas.area
fluxo = fotometria['aperture_sum'] - fundo_total

fluxo_seguro = np.maximum(fluxo, 1e-10) 
erro_fluxo = np.sqrt(fluxo_seguro + aberturas.area * (estat_anel.std**2))
snr = fluxo / erro_fluxo

coords = wcs.pixel_to_world(fontes['x_centroid'], fontes['y_centroid'])
df = pd.DataFrame({
    'ID': fontes['id'],
    'X_pix': fontes['x_centroid'],
    'Y_pix': fontes['y_centroid'],
    'RA_deg': coords.ra.deg,
    'Dec_deg': coords.dec.deg,
    'Fluxo': fluxo,
    'Erro_Fluxo': erro_fluxo,
    'SNR': snr,
    'Mag_Inst': -2.5 * np.log10(fluxo_seguro / exptime),
    'Erro_Mag': 1.0857 / snr,
    'Std_Fundo_Local': std_fundo,    # O valor std_fundo calculado logo após o bkg
    'Area_Ap': aberturas.area,       # A área em pixels da abertura circular
    'Exptime': exptime,
})


margem = np.ceil(RAIO_OUT)


df = df[
    (df['X_pix'] > margem) & (df['X_pix'] < largura - margem) &
    (df['Y_pix'] > margem) & (df['Y_pix'] < altura - margem) &
    (df['Fluxo'] > 0) &
    (df['SNR'] >= MIN_SNR)
].reset_index(drop=True)

if not df.empty:
    coords_pix = df[['X_pix', 'Y_pix']].values
    matriz_dist = distance.cdist(coords_pix, coords_pix)
    np.fill_diagonal(matriz_dist, np.inf)
    
   
    df = df[np.min(matriz_dist, axis=1) >= RAIO_OUT]


df = df.sort_values(by='Fluxo', ascending=False).reset_index(drop=True)


df.to_csv('fotometria_final_B.csv', index=False)

with open('regioes_aneis_B.reg', 'w') as f:
    f.write('global color=cyan width=1 select=1 edit=1 move=1 delete=1 include=1 source=1\nimage\n')
    for _, row in df.iterrows():
        # Removido o "-1" do texto do ID para sincronizar com o CSV
        f.write(f"circle({row['X_pix']+1:.2f},{row['Y_pix']+1:.2f},{RAIO_AP:.2f}) # color=cyan text={{{int(row['ID'])-0}}}\n") 
        f.write(f"annulus({row['X_pix']+1:.2f},{row['Y_pix']+1:.2f},{RAIO_IN:.2f},{RAIO_OUT:.2f}) # color=yellow\n")

print(f'Sucesso! {len(df)} fontes estelares limpas, brilhantes e isoladas foram salvas.')


