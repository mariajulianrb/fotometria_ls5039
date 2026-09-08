import matplotlib.pyplot as plt
import pandas as pd

# Carrega o arquivo final onde TODAS as estrelas receberam a calibração
df = pd.read_csv('fotometria_B_calibrada.csv')

plt.figure(figsize=(8, 5))
# Remove eventuais valores nulos ou infinitos causados por erros de extração
df = df[df['Mag_B_Calibrada'].notna()]

plt.hist(df['Mag_B_Calibrada'], bins=40, color='royalblue', edgecolor='black', alpha=0.8)
plt.axvline(x=18.45, color='purple', linestyle='--', linewidth=2, label='Limite 5-Sigma (18.45 mag)')

# Eixo invertido: magnitude menor (mais brilhante) à direita, maior (mais fraca) à esquerda
plt.gca().invert_xaxis() 

plt.title('Distribuição de Estrelas da Imagem vs Limite Teórico')
plt.xlabel('Magnitude Aparente Calibrada (B)')
plt.ylabel('Contagem de Estrelas Detectadas')
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()