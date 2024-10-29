#-------------------------------------------------------------------------------------------------------------
#                                                   HEA criteria
#-------------------------------------------------------------------------------------------------------------
"""
Cálculos para o critérios de High Entropy Alloys
1) Porcentagem atômica.
2) dHmix
3) atomic radii
4) electronegativity
5) VEC
6) Elastic-strain energy criterion

Elementos utilizados: 
elementos = [
    "Co", "Fe", "Ni", "Si", "Al", "Cr", "Mo", "Nb", "Ti", "C",
    "V", "Zr", "Mn", "Cu", "B", "Y", "Sn", "Li", "Mg", "Zn",
    "Sc", "Hf", "Ta", "W"
]

Outros arquivos:
- element_radii_electro_vec.csv
- Miedema_paired_dHmix.csv
- physical_values.csv
"""

# Importar bibliotecas
import re
import pandas as pd
import numpy as np
import itertools

#-------------------------------------------------------------------------------------------------------------
#### Nomenclatura de liga para porcentagem atômica
'''
--------------------------------------------------IMPORTANTE!--------------------------------------------------
Se houver parêntesis, deve começar com este, por exemplo:
(CoCrCuFeMnNiTiV)88.9Al11.1 - correto
Al11.1(CoCrCuFeMnNiTiV)88.9 - errado
'''

# Separar a nomenclatura entre com e sem parêntesis
def has_parentheses(string):
    return bool(re.search(r'\(.*?\)', string))

# Caso a escrita da liga não contenha "()"
def parse_alloy_1(alloy):
    # Regex para encontrar os elementos e suas quantidades
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
    matches = re.findall(pattern, alloy)
    
    total_atoms = 0
    element_dict = {}
    
    # Processa cada elemento encontrado
    for element, quantity in matches:
        quantity = float(quantity) if quantity else 1.0  # Considera 1 se não houver quantidade
        element_dict[element] = quantity
        total_atoms += quantity
    
    # Calcula porcentagem atômica
    for element in element_dict:
        element_dict[element] = (element_dict[element] / total_atoms) * 100
    
    return element_dict

# Caso a escrita da liga contenha "()"
def parse_alloy_2(alloy):
    # Verifica se a liga tem parênteses
    if '(' in alloy and ')' in alloy:
        # Captura a parte entre parênteses
        inner_part = alloy[alloy.index('(') + 1:alloy.index(')')]
        quantity_part = alloy[alloy.index(')') + 1:]

        # Pega o número total após os parênteses
        multiplier = float(re.search(r'\d+', quantity_part).group()) if re.search(r'\d+', quantity_part) else 1.0
        
        # Divisão equitativa dos elementos dentro dos parênteses
        inner_elements = re.findall(r'[A-Z][a-z]*', inner_part)
        num_inner_elements = len(inner_elements)

        # Cria o dicionário com as porcentagens
        result = {element: multiplier / num_inner_elements for element in inner_elements}

        # Adiciona os elementos fora dos parênteses
        outside_elements = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', quantity_part)
        for element, quantity in outside_elements:
            quantity = float(quantity) if quantity else 1.0
            result[element] = result.get(element, 0) + quantity

        return result

    return {}

# atomic_percentage()
def atomic_percentage(alloy):
    if has_parentheses(alloy):
        return parse_alloy_2(alloy)
    else:
        return parse_alloy_1(alloy)

#-------------------------------------------------------------------------------------------------------------
#### Cálculo de Entalpia de Mistura
'''
    Dados dos pares obtidos em Takeuchi_Materials Transactions, Vol 46, 12 (2005) 2817-2829_2005
    "Classification of Bulk Metallic Glasses by Atomic Size Difference, Heat of Mixing and Period of 
    Constituent Elements and Its Application to Characterization of the Main Alloying Element"
    doi: https://doi.org/10.2320/matertrans.46.2817

    obs.: Junto com esse módulo, baixe o arquivo: 'Miedema_paired_dHmix.csv'
'''

# dHmix
def dHmix(liga):

    # Colocar o path do arquivo "Miedema_paired_dHmix.csv"
    path_miedema = "Miedema_paired_dHmix.csv"

    # Miedema paired dHmix
    df = pd.read_csv(path_miedema)

    # Elementos da liga
    elements = atomic_percentage(liga).keys()

    # Pares de elementos
    alloy = pd.DataFrame(itertools.combinations(elements,2), columns=["Element_1", "Element_2"])

    lista = []
    for i in range(len(alloy)):
        lista.append((alloy["Element_1"][i], 
                    alloy["Element_2"][i], 
                    float(df[(df["Element_1"] == alloy["Element_1"][i])&(df["Element_2"] == alloy["Element_2"][i])]["paired_dHmix"].values)))

    df_alloy = pd.DataFrame(lista, columns = ["Element_1", "Element_2", "paired_dHmix"])

    # Adicionar porcentagem atômica
    df_alloy["%at element_1"] = df_alloy["Element_1"].map(atomic_percentage(liga))/100
    df_alloy["%at element_2"] = df_alloy["Element_2"].map(atomic_percentage(liga))/100

    # Fatores da soma de dHmix
    df_alloy["4*Hcicj"] = 4 * df_alloy["paired_dHmix"] * df_alloy["%at element_1"] * df_alloy["%at element_2"]

    # Retornar
    return df_alloy["4*Hcicj"].sum()

#-------------------------------------------------------------------------------------------------------------
#### Atomic radius difference [%]
"""
The data on 'element_radii_electro_vec.csv' comes from Guo et al. Phase stability in high entropy alloys:
Formation of solid-solution phase or amourphous phase. Progress in Natural Science: Materials International
21 (2011) 433-446. doi: 10.1016/S1002-0071(12)60080-X
"""
def atomic_radii(alloy):
    # -----------------------------------------------------------
    # Radius, Pauling_Electronegativity, VEC data
    df = pd.read_csv("element_radii_electro_vec.csv")
    #alloy = "Al0.3NbTa0.8Ti1.4V0.2Zr1.3"
    dr = pd.DataFrame(atomic_percentage(alloy).keys(), columns=["Elementos"])
    # Para cada elemento, criar uma nova coluna com map da concentração química em %at.
    dr["atomic_perc"] =dr["Elementos"].map(atomic_percentage(alloy))/100
    # Usar a tabela element_radii_electro_vec.csv para coletar os dados
    dr = dr.merge(df, left_on="Elementos", right_on= "Symbol", how='left').drop(columns=["Symbol"])
    # -----------------------------------------------------------

    # average radius
    avg_rad = np.sum(dr["atomic_perc"]*dr["Radius (Å)"])
    # Diff radii
    rad_list = dr["atomic_perc"] * (1 - (dr["Radius (Å)"] / avg_rad))**2
    diff_rad = 100*np.sqrt(np.sum(rad_list))

    # Return
    return diff_rad

#-------------------------------------------------------------------------------------------------------------
#### Pauling Electronegativity
"""
The data on 'element_radii_electro_vec.csv' comes from Guo et al. Phase stability in high entropy alloys:
Formation of solid-solution phase or amourphous phase. Progress in Natural Science: Materials International
21 (2011) 433-446. doi: 10.1016/S1002-0071(12)60080-X  
"""
def electronegativity(alloy):
    # -----------------------------------------------------------
    # Radius, Pauling_Electronegativity, VEC data
    df = pd.read_csv("element_radii_electro_vec.csv")
    #alloy = "Al0.3NbTa0.8Ti1.4V0.2Zr1.3"
    dr = pd.DataFrame(atomic_percentage(alloy).keys(), columns=["Elementos"])
    # Para cada elemento, criar uma nova coluna com map da concentração química em %at.
    dr["atomic_perc"] =dr["Elementos"].map(atomic_percentage(alloy))/100
    # Usar a tabela element_radii_electro_vec.csv para coletar os dados
    dr = dr.merge(df, left_on="Elementos", right_on= "Symbol", how='left').drop(columns=["Symbol"])
    # -----------------------------------------------------------

    # average electronegativity
    avg_electro = sum(dr["atomic_perc"]*dr["Pauling Electronegativity"])
    # Electronegativity criteria
    electro_list = dr["atomic_perc"] * (dr["Pauling Electronegativity"] - avg_electro)**2
    diff_electro = np.sqrt(sum(electro_list))

    # Return
    return diff_electro

#-------------------------------------------------------------------------------------------------------------
#### VEC
"""
The data on 'element_radii_electro_vec.csv' comes from Guo et al. Phase stability in high entropy alloys:
Formation of solid-solution phase or amourphous phase. Progress in Natural Science: Materials International
21 (2011) 433-446. doi: 10.1016/S1002-0071(12)60080-X  
"""
def vec(alloy):
    # -----------------------------------------------------------
    # Radius, Pauling_Electronegativity, VEC data
    df = pd.read_csv("element_radii_electro_vec.csv")
    # alloy
    dr = pd.DataFrame(atomic_percentage(alloy).keys(), columns=["Elementos"])
    # Para cada elemento, criar uma nova coluna com map da concentração química em %at.
    dr["atomic_perc"] =dr["Elementos"].map(atomic_percentage(alloy))/100
    dr = dr.merge(df, left_on="Elementos", right_on= "Symbol", how='left').drop(columns=["Symbol"])
    # -----------------------------------------------------------

    # VEC criteria
    vec = np.sum(dr["atomic_perc"]*dr["VEC"])

    # Return
    return vec
#-------------------------------------------------------------------------------------------------------------
#### Entropy of Mixture
"""
Usa-se a função atomic_percentage(alloy) 
"""
def dSmix(alloy):

    import numpy as np

    # Gas constant : R = 8.3145 J.mol^-1.K^-1 
    R = 8.3145
    # -----------------------------------------------------------
    # Radius, Pauling_Electronegativity, VEC data
    df = pd.read_csv("element_radii_electro_vec.csv")
    # alloy table
    dr = pd.DataFrame(atomic_percentage(alloy).keys(), columns=["Elementos"])
    # Para cada elemento, criar uma nova coluna com map da concentração química em %at.
    dr["atomic_perc"] =dr["Elementos"].map(atomic_percentage(alloy))/100
    dr = dr.merge(df, left_on="Elementos", right_on= "Symbol", how='left').drop(columns=["Symbol"])
    # -----------------------------------------------------------

    # Entropy of mixture
    entropy = -R*sum(dr["atomic_perc"]*np.log(dr["atomic_perc"]))

    # Return
    return entropy
#-------------------------------------------------------------------------------------------------------------
#### Elastic-strain energy criterion
"""
Angelo F. Andreoli et al. The elastic-strain energy criterion of phase formation for complex concentrated alloys.
Materialia 5 (2019) 100222. doi: 10.1016/j.mtla.2019.100222

Dados de Bulk modulus foi retirado de :
Ref [40] The photographic periodic table of the elements. http://periodictable.com/index.html.
Atomic volume of the element i, is calculated as : 
    atomic_volume = (4/3) * pi() * atomic_radii^3 * 6.02 * 10^23 * 10^-30[m³]
"""
def dHel(alloy):

    # Um dataframe para cada Liga
    df2 = pd.read_csv('physical_values.csv')
    dr2 = pd.DataFrame(atomic_percentage(alloy).keys(), columns=["Elementos"])
    dr2["Atomic_perc"] = dr2["Elementos"].map(atomic_percentage(alloy))/100
    dr2 = dr2.merge(df2, on="Elementos", how="left")

    # V calculation
    numerator = np.sum(dr2["Atomic_perc"]*dr2["Bulk modulus (GPa)"]*(10**9)*dr2["Atomic volume"]*(10**-6))
    denominator = np.sum(dr2["Atomic_perc"]*dr2["Bulk modulus (GPa)"]*(10**9))
    V = numerator/denominator
    # dHel
    dHel = np.sum(dr2["Atomic_perc"]*dr2["Bulk modulus (GPa)"]*(10**9)*(((dr2["Atomic volume"]*(10**-6))-V)**2)/(2*(dr2["Atomic volume"]*(10**-6))))

    return dHel/1000
#-------------------------------------------------------------------------------------------------------------