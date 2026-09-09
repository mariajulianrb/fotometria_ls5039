import matplotlib.pyplot as plt
import pandas as pd

# Certifique-se de que o nome do arquivo bate com o gerado no script anterior
ARQUIVO_CSV = 'fotometria_B_calibrada_Gaia.csv' 
LIMITE_CALCULADO = 18.05

df = pd.read_csv(ARQUIVO_CSV)
df = df[df['Mag_B_Calibrada'].notna()]

plt.figure(figsize=(8, 5))

plt.hist(df['Mag_B_Calibrada'], bins=40, color='royalblue', edgecolor='black', alpha=0.8)
plt.axvline(x=LIMITE_CALCULADO, color='purple', linestyle='--', linewidth=2, label=f'Limite 5-Sigma ({LIMITE_CALCULADO} mag)')

# Eixo invertido: magnitude menor (mais brilhante) à direita, maior (mais fraca) à esquerda
plt.gca().invert_xaxis() 

plt.title('Distribuição de Estrelas da Imagem vs Limite Teórico')
plt.xlabel('Magnitude Aparente Calibrada (B)')
plt.ylabel('Contagem de Estrelas Detectadas')
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()