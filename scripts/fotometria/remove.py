import pandas as pd

# Parâmetros das aberturas
FWHM = 8.89
raio_abertura = 2.0 * FWHM
raio_in = 3.0 * FWHM
raio_out = 4.0 * FWHM

# 1. Carregar a tabela gerada
df = pd.read_csv('fotometria_final_B.csv')

# 2. Lista de IDs identificados na vinhetagem
ids_vinhetagem = [787, 776, 753]

# 3. Remover os IDs e salvar a nova tabela
df_limpo = df[~df['ID'].isin(ids_vinhetagem)].reset_index(drop=True)
df_limpo.to_csv('fotometria_limpa_B.csv', index=False)

# 4. Gerar o novo arquivo de regiões (mantém o regioes_aneis_B.reg intocado)
with open('regioes_limpas_B.reg', 'w') as f:
    f.write(
        'global color=cyan width=1 select=1 edit=1 move=1 delete=1 include=1'
        ' source=1\nimage\n'
    )
    for id_est, x, y in zip(
        df_limpo['ID'], df_limpo['X_pix'], df_limpo['Y_pix']
    ):
        f.write(
            f'circle({x+1:.2f},{y+1:.2f},{raio_abertura:.2f}) # color=cyan'
            f' text={{{id_est}}}\n'
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