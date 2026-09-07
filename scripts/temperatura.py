import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# 1. Função da Relação de Ballesteros (2012) para T_eff via (B - V)
def estimar_temperatura_ballesteros(cor_B_V):
    return 4600 * (1 / (0.92 * cor_B_V + 1.7) + 1 / (0.92 * cor_B_V + 0.62))

# 2. Carregar o arquivo calibrado
# Altere o nome caso seu arquivo B-R calibrado tenha outro nome
ARQUIVO_DADOS = 'resultado_fotometria_calibrada_BR.csv' 

try:
    df_obs = pd.read_csv(ARQUIVO_DADOS)
except FileNotFoundError:
    print(f"Erro: Arquivo '{ARQUIVO_DADOS}' não encontrado na pasta atual.")
    exit()

# 3. Localizar a LS 5039 pelas Coordenadas Celestes (RA: 18h 26m 15.06s, Dec: -14° 50′ 54.2″)
coord_ls5039 = SkyCoord(ra=276.56276*u.deg, dec=-14.84838*u.deg)
coords_df = SkyCoord(ra=df_obs['RA_deg'].values*u.deg, dec=df_obs['Dec_deg'].values*u.deg)

idx_alvo, sep2d, _ = coord_ls5039.match_to_catalog_sky(coords_df)

if sep2d > 3.0 * u.arcsec:
    print("Aviso: LS 5039 não foi encontrada no catálogo com precisão < 3 arcsec.")
    exit()

# 4. Extrair os valores observados para o alvo
cor_obs_BR = df_obs['Cor_B_R'].iloc[idx_alvo]
mag_R_obs = df_obs['Mag_Ap_R'].iloc[idx_alvo]

# Conversão empírica aproximada de (B - R) para (B - V) no sistema Johnson-Cousins
# Em geral: (B - V) ≈ 0.62 * (B - R)
cor_obs_BV = 0.62 * cor_obs_BR

# 5. Cálculo da Temperatura Aparente (Sem Correção de Poeira)
T_antes = estimar_temperatura_ballesteros(cor_obs_BV)

# 6. Correção de Extinção (Valores Teóricos da Literatura)
# Cor intrínseca para uma estrela O6.5V: (B - R)_0 = -0.46 mag -> (B - V)_0 = -0.31 mag
cor_intrinseca_BV = -0.31
excesso_BV = cor_obs_BV - cor_intrinseca_BV

# 7. Cálculo da Temperatura Real (Com Correção de Poeira)
T_depois = estimar_temperatura_ballesteros(cor_intrinseca_BV)

# 8. Impressão dos Resultados no Terminal
print("\n" + "="*55)
print("      ANÁLISE DE TEMPERATURA DA LS 5039 (T_eff)")
print("="*55)
print(f"Cor Observada Medida (B - R):        {cor_obs_BR:.3f} mag")
print(f"Cor Aparenta Estimada (B - V)_obs:  {cor_obs_BV:.3f} mag")
print(f"Temperatura Aparente (SEM CORREÇÃO): {T_antes:.0f} K")
print("-" * 55)
print(f"Cor Intrínseca Teórica (B - V)_0:    {cor_intrinseca_BV:.3f} mag (O6.5V)")
print(f"Excesso de Cor Derivado E(B - V):    {excesso_BV:.3f} mag")
print(f"Temperatura Real (COM CORREÇÃO):     {T_depois:.0f} K")
print("="*55 + "\n")