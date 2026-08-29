import glob
import numpy as np
from astropy.io import fits

bias_files = sorted(glob.glob('bias_*.fits'))
print(f"Combinando {len(bias_files)} arquivos de Bias...")

bias_stack = [fits.getdata(f) for f in bias_files]

master_bias = np.median(bias_stack, axis=0)

header = fits.getheader(bias_files[0])
fits.writeto('master_bias.fits', master_bias, header=header, overwrite=True)

print("Master Bias gerado com sucesso: master_bias.fits")


# ds9 master_bias.fits &
# python -c "from astropy.io import fits; import numpy as np; b1 = fits.getdata('bias_001.fits'); mb = fits.getdata('master_bias.fits'); print(f'Ruído Bias 1: {np.std(b1):.2f} ADU'); print(f'Ruído Master Bias: {np.std(mb):.2f} ADU')"