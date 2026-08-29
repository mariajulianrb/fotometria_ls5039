import glob
import numpy as np
from astropy.io import fits

exposure_times = ['12', '23', '25', '50']

for exp in exposure_times:
    dark_files = sorted(glob.glob(f'dark_{exp}_*.fits'))
    
    if not dark_files:
        print(f"Nenhum arquivo encontrado para {exp}s.")
        continue
        
    print(f"Combinando {len(dark_files)} arquivos de Dark ({exp}s)...")
    
    dark_stack = [fits.getdata(f) for f in dark_files]
    
    master_dark = np.median(dark_stack, axis=0)
    
    header = fits.getheader(dark_files[0])
    
    output_name = f'master_dark_{exp}s.fits'
    fits.writeto(output_name, master_dark, header=header, overwrite=True)
    print(f"✓ {output_name} gerado com sucesso!")

print("Master Darks concluídos!")