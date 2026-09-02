# --- ADICIONAR NO FINAL DO SCRIPT ---

# Salvar arquivo de regiões para o DS9
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

print(f"Arquivo de regiões '{nome_reg}' gerado com sucesso!")