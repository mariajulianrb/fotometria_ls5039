import glob
import numpy as np
from astropy.io import fits

filters = ['b', 'g', 'r']
resumo_medias = {}

for band in filters:
    flat_files = sorted(glob.glob(f'flat_{band}_[0-9]*.fits'))
    
    if not flat_files:
        print(f"Nenhum arquivo encontrado para o filtro {band.upper()}.")
        continue
        
    print(f"--- Processando Filtro {band.upper()} ({len(flat_files)} arquivos) ---")
    

    flat_stack = [fits.getdata(f) for f in flat_files]
    

    master_flat = np.median(flat_stack, axis=0)
    
 
    mean_value = np.mean(master_flat)
    std_value = np.std(master_flat)
    resumo_medias[band.upper()] = mean_value
    
    print(f"  Média  (imstat) : {mean_value:.4f} ADU")
    print(f"  StdDev (imstat) : {std_value:.4f} ADU")
    
    flat_normalized = master_flat / mean_value
    
    header = fits.getheader(flat_files[0])
    output_name = f'master_flat_{band}_norm.fits'
    fits.writeto(output_name, flat_normalized, header=header, overwrite=True)
    print(f"✓ Arquivo gerado: {output_name}\n")

print("=" * 45)
print("  RESUMO DAS MÉDIAS (VALORES DE NORMALIZAÇÃO)")
print("=" * 45)
for band, media in resumo_medias.items():
    print(f"  Filtro {band}: {media:.4f} ADU")
print("=" * 45)