import pandas as pd

# Parâmetros das aberturas
FWHM = 8.89
raio_abertura = 1.5 * FWHM
raio_in = 3.0 * FWHM
raio_out = 4.0 * FWHM

# 1. Carregar a tabela gerada
df = pd.read_csv('fotometria_final_B.csv')

# 2. Lista de IDs identificados na vinhetagem
ids_vinhetagem = [
    841,
    828,
    814,
    803,
    765,
    819,
    812,
    833,
    802,
    783,
    771,
    784,
    216,
    161,
    152,
    129,
    106,
    89,
    67,
    58,
    45,
    15,
    22,
    34,
    143,
    128,
    101,
    35,
    43,
    31,
    42,
    59,
    79,
    75,
    94,
    496,
    531,
    372,
    568,
    644,
    712,
    177,
    153,
    132,
    164,
    132,

]

# 3. Remover os IDs e salvar a nova tabela
df_limpo = df[~df['ID'].isin(ids_vinhetagem)].reset_index(drop=True)
df_limpo.to_csv('fotometria_limpa_B.csv', index=False)

# 4. Gerar o novo arquivo de regiões sem os textos impresso no DS9
with open('regioes_limpas_B.reg', 'w') as f:
    f.write(
        'global color=cyan width=1 select=1 edit=1 move=1 delete=1 include=1'
        ' source=1\nimage\n'
    )
    for x, y in zip(df_limpo['X_pix'], df_limpo['Y_pix']):
        f.write(
            f'circle({x+1:.2f},{y+1:.2f},{raio_abertura:.2f}) # color=cyan\n'
        )
        f.write(
            f'annulus({x+1:.2f},{y+1:.2f},{raio_in:.2f},{raio_out:.2f}) #'
            ' color=yellow\n'
        )

print(
    f'Removidas {len(ids_vinhetagem)} estrelas da vinhetagem. Restam'
    f' {len(df_limpo)} fontes.'
)
print("Arquivo 'regioes_limpas_B.reg' criado com sucesso!")