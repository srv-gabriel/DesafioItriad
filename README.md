# DesafioItriad

O objetivo do desafio é desenvolver um modelo capaz de predizer o valor de um determinado imóvel baseado em suas características. O conjunto de dados inicial apresenta 80 características mais uma coluna com o valor de venda do imóvel, que é a coluna-alvo.

## Tratamento dos dados

O conjunto apresenta alguns atributos com valores faltantes e agluns atributos com valores nominais, os quais necessitam ser tratados antes de iniciar qualquer predição. O primeiro passo tomado na solução desenvolvida é tratar os valores faltantes

### Missing Values

Primeiro foram removidos atributos com uma quantidade grande de valores faltantes. As colunas removidas e seus respectivos números de míssing values são as seguintes:

 - 'MiscFeature'  : 1406 instâncias sem valores
 - 'Fence'        : 1179 instâncias sem valores
 - 'PoolQC'       : 1453 instâncias sem valores
 - 'Alley'        : 1369 instâncias sem valores
 - 'FireplaceQu'  : 690  instâncias sem valores
 
Após a remoção, foi feito o processo de imputação nos outros atributos. Veja que, devido a quantidade de valores faltantes nos atributos removidos ser muito grande, aplicar imputação neles geraria um grande viés no nosso conjunto de dados, dessa forma primeiro foi feita a remoção desses atributos e então aplicamos imputação nos valores faltantes restante. Os atributos que receberam imputação foram os seguintes:

**No conjunto de treino:**
- 'LotFrontage'   : 259 instâncias - Utilizando média
- 'BsmtQual'      : 
- 'BsmtCond'      :
- 'BsmtExposure'  :
- 'BsmtFinType1'  :
- 'BsmtFinType2'  :
- 'Electrical'    :
- 'GarageType'    :
- 'GarageYrBlt'   :
- 'GarageFinish'  :
- 'GarageQual'    :
- 'GarageCond'    :
- 'MasVnrType'    :
- 'MasVnrArea'    :

**No conjunto de test:**
- 'LotFrontage'   : 259 instâncias - Utilizando média
- 'MSZoning'      :
- 'Utilities'     :
- 'Exterior1st'   :
- 'Exterior2nd'   :
- 'BsmtQual'      : 
- 'BsmtCond'      :
- 'BsmtExposure'  :
- 'BsmtFinType1'  :
- 'BsmtFinSF1'    :
- 'BsmtFinType2'  :
- 'BsmtFinSF2'    :
- 'BsmtUnfSF'     :
- 'TotalBsmtSF'   :
- 'BsmtFullBath'  :
- 'BsmtHalfBath'  :
- 'KitchenQual'   :
- 'Functional'    :
- 'GarageType'    :
- 'GarageYrBlt'   :
- 'GarageFinish'  :
- 'GarageCars'    :
- 'GarageArea'    :
- 'GarageQual'    :
- 'GarageCond'    :
- 'MasVnrType'    :
- 'MasVnrArea'    :
- 'SaleType'      :

### Tratando Features nominais

Para atributos nominais foi aplicada a numerização, que consiste em substituir as classes de um atributo por números. Esse passo é necessário para que o treino pelos modelos de aprendizado de máquina seja possível.

**Atributos modificados:**
'MSZoning'      :
'Street'        :
'LotShape'      :
'LandContour'   :
'Utilities'     :
'LotConfig'     :
'LandSlope'     :
'Neighborhood'  :
'Condition1'    :
'Condition2'    :
'BldgType'      :
'HouseStyle'    :
'RoofStyle'     :
'RoofMatl'      :
'Exterior1st'   :
'Exterior2nd'   :
'MasVnrType'    :
'ExterQual'     :
'ExterCond'     :
'Foundation'    :
'BsmtQual'      :
'BsmtCond'      :
'BsmtExposure'  :
'BsmtFinType1'  :
'BsmtFinType2'  :
'Heating'       :
'HeatingQC'     :
'CentralAir'    :
'Electrical'    :
'KitchenQual'   :
'Functional'    :
'GarageType'    :
'GarageFinish'  :
'GarageQual'    :
'GarageCond'    :
'PavedDrive'    :
'SaleType'      :
'SaleCondition' :

### Divisão dos dados

Para podermos analisar e melhorar os parâmetros utilizados no regressor, os dados serão separados em conjunto de treino e validação, sendo o conjunto de validação utilizado para fazermos um _tunning_ no modelo. Essa separação é necessária para verificarmos o tradeoff viés-variância antes de aplicarmos o modelo no conjunto de testes
