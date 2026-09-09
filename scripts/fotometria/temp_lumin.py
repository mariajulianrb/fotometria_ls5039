import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

# ==========================================
# 1. PARÂMETROS FÍSICOS E DA LITERATURA
# ==========================================
ARQUIVO_FOTOMETRIA = 'resultado_fotometria_calibrada_BG.csv'
RA_ALVO, DEC_ALVO = 276.5627, -14.8479

# Parâmetros de Reddening para o campo de LS 5039
E_BG = 1.28              # Excesso de cor intrínseco E(B-G)
A_G = 3.07               # Extinção interestelar na banda G
MAG_ABS_SOL_G = 4.66     # Magnitude absoluta do Sol na banda G

# Tabela empírica de Pecaut & Mamajek (2013) para Sequência Principal
# Relaciona o índice de cor intrínseco (B-G)_0 diretamente à Temperatura Efetiva (K)
BG0_GRID = np.array([-0.363, -0.352, -0.330, -0.220, -0.110, 0.000, 0.165, 0.330, 0.550, 0.715, 0.880, 1.100, 1.320, 1.551, 1.760])
TEFF_GRID = np.array([42000, 39000, 31400, 20000, 14000, 9500, 8000, 7000, 6100, 5770, 5300, 4800, 4300, 3850, 3300])

def estimar_temperatura(cor_BG_0):
    """Retorna T_eff via interpolação linear dos dados empíricos estelares."""
    return np.interp(cor_BG_0, BG0_GRID, TEFF_GRID)

# ==========================================
# 2. CARREGAMENTO DOS DADOS FOTOMÉTRICOS
# ==========================================
df_obs = pd.read_csv(ARQUIVO_FOTOMETRIA)
coords_obs = SkyCoord(ra=df_obs['RA_deg'].values * u.deg, dec=df_obs['Dec_deg'].values * u.deg)

# ==========================================
# 3. CRUZAMENTO ASTROMÉTRICO COM O GAIA DR3
# ==========================================
print("Baixando paralaxes do Gaia DR3 via VizieR...")
v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Plx', 'e_Plx'], row_limit=-1)
coord_centro = SkyCoord(ra=RA_ALVO * u.deg, dec=DEC_ALVO * u.deg)
tabela_gaia = v.query_region(coord_centro, radius=15 * u.arcmin, catalog='I/355/gaiadr3')[0].to_pandas()

# Filtro de qualidade astrométrica (Sinal/Ruído da paralaxe > 3)
tabela_gaia = tabela_gaia[(tabela_gaia['Plx'] > 0) & (tabela_gaia['Plx'] / tabela_gaia['e_Plx'] > 3)]
coords_gaia = SkyCoord(ra=tabela_gaia['RA_ICRS'].values * u.deg, dec=tabela_gaia['DE_ICRS'].values * u.deg)

# Identificação das estrelas em comum (raio de tolerância de 2.5")
idx_gaia, d2d_gaia, _ = coords_obs.match_to_catalog_sky(coords_gaia)
valid_gaia = d2d_gaia < (2.5 * u.arcsec)

df = df_obs[valid_gaia].copy().reset_index(drop=True)
df['Paralaxe_mas'] = tabela_gaia['Plx'].iloc[idx_gaia[valid_gaia]].values

# ==========================================
# 4. CÁLCULO DOS PARÂMETROS INTRÍNSECOS
# ==========================================
# Cor Intrinseca e Temperatura
df['Cor_BG_0'] = df['Cor_B_G'] - E_BG
df['T_eff'] = estimar_temperatura(df['Cor_BG_0'])

# Distância (pc) e Módulo de Distância
df['Distancia_pc'] = 1000.0 / df['Paralaxe_mas']
modulo_distancia = 5 * np.log10(df['Distancia_pc']) - 5

# Magnitude Absoluta Desavermelhada e Luminosidade Solar
df['Mag_Abs_G0'] = df['Mag_Ap_G'] - modulo_distancia - A_G
df['Luminosidade'] = 10 ** (-0.4 * (df['Mag_Abs_G0'] - MAG_ABS_SOL_G))

# Identificação exata da LS 5039 (menor separação do centro coordenado)
coords_fisico = SkyCoord(ra=df['RA_deg'].values * u.deg, dec=df['Dec_deg'].values * u.deg)
idx_alvo = np.argmin(coords_fisico.separation(coord_centro))

print(f"Estrelas processadas: {len(df)}")
print(f"LS 5039 (Alvo) -> T_eff: {df['T_eff'].iloc[idx_alvo]:.0f} K | Luminosidade: {df['Luminosidade'].iloc[idx_alvo]:.1f} L_sol")

# ==========================================
# 5. CONSTRUÇÃO DO DIAGRAMA HR
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

# Fundo estilizado para as estrelas de campo (Coloridas por T_eff)
sc = ax.scatter(
    df['T_eff'], df['Luminosidade'],
    c=df['T_eff'], cmap='RdYlBu', # Mapa de cores astrofísico (Azul = Quente, Vermelho = Frio)
    s=40, alpha=0.8, edgecolor='black', linewidth=0.4,
    label='Estrelas de Campo'
)

# Destaque robusto para LS 5039
ax.scatter(
    df['T_eff'].iloc[idx_alvo], df['Luminosidade'].iloc[idx_alvo],
    color='cyan', marker='*', s=450, edgecolor='black', linewidth=1.5,
    zorder=10, label=f'LS 5039 ($T_{{eff}} \\approx {df["T_eff"].iloc[idx_alvo]:.0f}$ K)'
)

# Linha Base de Luminosidade Solar
ax.axhline(1.Como você não enviou o código anterior na nossa conversa atual, criei uma versão definitiva e independente utilizando o **Plotly** em Python. Para visualização astronômica, o Plotly representa o "melhor dos mundos" porque gera um gráfico 3D **interativo**: você pode girar, dar zoom e explorar a nuvem de estrelas, sem perder o destaque da LS 5039.

Este código simula um campo estelar denso com fundo escuro (estilo espaço profundo) e isola a LS 5039 com um marcador de diamante vermelho, contorno amarelo e texto fixo.

```python
import plotly.graph_objects as go
import numpy as np

# 1. Gerando dados simulados para a "maioria das estrelas"
np.random.seed(42) # Para manter o padrão estelar consistente
n_estrelas = 1500

# Distribuindo estrelas em um espaço 3D
x_fundo = np.random.normal(0, 50, n_estrelas)
y_fundo = np.random.normal(0, 50, n_estrelas)
z_fundo = np.random.normal(0, 50, n_estrelas)

# Variando o tamanho para simular diferentes magnitudes (brilhos)
tamanhos_fundo = np.random.uniform(1, 3.5, n_estrelas)

# Iniciar a figura
fig = go.Figure()

# 2. Plotando as estrelas de fundo
fig.add_trace(go.Scatter3d(
    x=x_fundo, 
    y=y_fundo, 
    z=z_fundo,
    mode='markers',
    marker=dict(
        size=tamanhos_fundo,
        color=z_fundo, # Gradiente de cor baseado na profundidade
        colorscale='Plotly3', # Escala de cores vibrante, boa para fundos escuros
        opacity=0.7
    ),
    name='Campo Estelar',
    hoverinfo='none' # Desativa o hover no fundo para focar na LS 5039
))

# 3. Plotando a estrela LS 5039 em DESTAQUE ABSOLUTO
# Coordenadas relativas posicionadas para fácil visualização
x_ls, y_ls, z_ls = [10], [10], [15] 

fig.add_trace(go.Scatter3d(
    x=x_ls, 
    y=y_ls, 
    z=z_ls,
    mode='markers+text',
    marker=dict(
        size=14,
        color='red',
        symbol='diamond', # Formato distinto
        line=dict(color='yellow', width=2) # Contorno brilhante
    ),
    text=['⭐ LS 5039'],
    textposition='top center',
    textfont=dict(color='yellow', size=16, family='Arial Black'),
    name='Microquasar LS 5039',
    hovertemplate='<b>LS 5039</b><br>Sistema Binário de Alta Massa<extra></extra>'
))

# 4. Configurando o "Melhor dos Mundos" (Visual Limpo e Fundo Espacial)
fig.update_layout(
    title=dict(
        text='Mapa Estelar Interativo 3D: Destaque para LS 5039', 
        font=dict(color='white', size=22),
        x=0.5 # Centraliza o título
    ),
    paper_bgcolor='black', # Fundo externo preto
    scene=dict(
        bgcolor='black', # Fundo interno (espaço) preto
        xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5) # Zoom inicial
        )
    ),
    margin=dict(l=0, r=0, b=0, t=60),
    showlegend=True,
    legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0)')
)

# Mostrar o gráfico interativo
fig.show()