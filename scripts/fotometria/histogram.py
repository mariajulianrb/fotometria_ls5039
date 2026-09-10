import matplotlib.pyplot as plt
import pandas as pd

ARQUIVO_CSV = 'fotometria_R_calibrada_Gaia.csv' 
LIMITE_CALCULADO = 17.25  # Substitua pelo valor impresso no terminal pelo script principal

df = pd.read_csv(ARQUIVO_CSV)
df = df[df['Mag_R_Calibrada'].notna()]

plt.figure(figsize=(8, 5))

plt.hist(df['Mag_R_Calibrada'], bins=40, color='crimson', edgecolor='black', alpha=0.75, label='Fontes Extraídas (R)')
plt.axvline(x=LIMITE_CALCULADO, color='purple', linestyle='--', linewidth=2, label=f'Limite 5-$\sigma$ ({LIMITE_CALCULADO:.2f} mag)')

# Convenção astronômica: estrelas mais brilhantes (menor magnitude) à esquerda ou invertido
# Caso queira a convenção padrão da física/astronomia (brilhantes à esquerda, fracas à direita):
# plt.gca().invert_xaxis() 

plt.title('Distribuição de Magnitudes e Profundidade (Filtro R)', fontsize=12)
plt.xlabel('Magnitude Aparente Calibrada ($R$)', fontsize=11)
plt.ylabel('Número de Estrelas (N)', fontsize=11)
plt.legend(frameon=True, facecolor='white', edgecolor='gray')
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()

plt.savefig('histograma_profundidade_R.png', dpi=300)
plt.show()