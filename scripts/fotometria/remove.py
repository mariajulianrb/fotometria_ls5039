import pandas as pd

FWHM = 7.51846 
raio_abertura = 1.5 * FWHM
raio_in = 3.0 * FWHM
raio_out = 4.0 * FWHM

df = pd.read_csv('fotometria_abertura_R.csv')

ids_vinhetagem = [
    1797,
    1774,
    1763,
    118,
    39,
    92,
    122,
    197,
    228,
    337,
    427,
    394,
    430,
    11,
    32,
    64,
    134,
    123,
    1703,
    1660,
    1499,


]

"""R

    1107,
    1095,
    1088,
    65,
    25,
    54,
    270,
    248,
    268,
    115,
    134,
    9,
    21,
    35,
    78,
    70,
    85,
    119,"""

"""G
    374,
    147,
    323,
    201,
    143,
    671,
    2641,
    2608,
    2566,
    2491,
    2433,
    2385,
    197,
    109,
    76
    2600"""


"""B
    86,
    72,
    56,
    41,
    30,
    6,
    44,
    130,
    111,
    94,
    72,
    66,
    34,
    12,
    123,
    712,
    703,
    681,
    649,
    642,
    680,
    690,
    664,
    694,
    662,
    604,
    188,
    163,
    147,
    33,
    21,
    120,
    60,
"""

df_limpo = df[~df['ID'].isin(ids_vinhetagem)].reset_index(drop=True)
df_limpo.to_csv('fotometria_abertura_R.csv', index=False)

with open('regioes_limpas_R.reg', 'w') as f:
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
print("Arquivo 'regioes_limpas_R.reg' criado com sucesso!")