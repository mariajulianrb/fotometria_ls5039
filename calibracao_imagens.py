import glob
import os
from astropy.io import fits

master_bias = fits.getdata('master_bias.fits')

darks = {}
for d_file in glob.glob('master_dark_*s.fits'):
    exp_str = d_file.split('_')[-1].replace('s.fits', '')
    darks[exp_str] = fits.getdata(d_file) - master_bias

flats = {
    'b': fits.getdata('master_flat_b_norm.fits'),
    'g': fits.getdata('master_flat_g_norm.fits'),
    'r': fits.getdata('master_flat_r_norm.fits')
}

sci_files = sorted([
    f for f in glob.glob('ls5039_*.fits') 
    if not f.startswith('calib_')
])

print(f"Iniciando calibração de {len(sci_files)} imagens...\n")

for filepath in sci_files:
    header = fits.getheader(filepath)
    raw_data = fits.getdata(filepath)
    
    exptime = str(int(header.get('EXPTIME', 0)))
    
    filtro = None
    for b in ['b', 'g', 'r']:
        if f'_{b}_' in filepath.lower():
            filtro = b
            break
    if not filtro:
        filtro = str(header.get('FILTER', '')).strip().lower()

    if exptime not in darks:
        print(f"⚠️ Dark de {exptime}s não encontrado. Pulando {filepath}...")
        continue
    if filtro not in flats:
        print(f"⚠️ Flat da banda '{filtro}' não encontrado. Pulando {filepath}...")
        continue

    # I = (Iraw - Mbias - Mdark) / flatn
    mdark = darks[exptime]
    flatn = flats[filtro]
    
    calib_data = (raw_data - master_bias - mdark) / flatn

    # Salvar imagem final
    out_name = f"calib_{os.path.basename(filepath)}"
    fits.writeto(out_name, calib_data, header=header, overwrite=True)
    print(f"✓ Calibrada: {out_name} (Exp: {exptime}s | Filtro: {filtro.upper()})")

print("\nProcesso concluído!")