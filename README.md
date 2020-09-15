# DesafioItriad

O objetivo do desafio é desenvolver um modelo capaz de predizer o valor de um determinado imóvel baseado em suas características. O conjunto de dados inicial apresenta 80 características mais uma coluna com o valor de venda do imóvel, que é a coluna-alvo.

## Bibliotecas Utilizadas

 - Scikit-learn   : O scikit-learn, ou sklearn, é uma biblioteca de código aberto que possui diversos métodos que são frequentemente utilizados em projetos de machine learning. Nesse desafio, os principais recursos utilizados dessa biblioteca foram os algoritmos clássicos de machine learning (SVM e Random Forest) e as métricas utilizadas para avaliar os modelos
 - Matplotlib     : Biblioteca utilizada para visualização gráfica dos dados
 - Pandas         : Principal biblioteca utilizada para o processamento de dados. Com ela foi feita todo a visualização dos dados em forma de tabelas e o tratamento de dados apresentado na seção seguinte
 - Tensorflow 2.0 : Tensorflow 2.0 utilizando Keras em seu núcleo é uma das bibliotecas mais utilizadas para o desenvolvimento de Deep Learning. Com ela é possível desenvolver redes neurais de forma prática.
 - OS             : Biblioteca utilizada para definir os paths de onde os datasets vão ser carregados e onde o resultado será salvo
 - Numpy          : Biblioteca matemática muito utilizada, faz operações matemáticas de forma eficiente e paralela

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
- 'BsmtQual'      : 37  instâncias - Utilizando moda
- 'BsmtCond'      : 37  instâncias - Utilizando moda
- 'BsmtExposure'  : 38  instâncias - Utilizando moda
- 'BsmtFinType1'  : 37  instâncias - Utilizando moda
- 'BsmtFinType2'  : 38  instâncias - Utilizando moda
- 'Electrical'    : 1   instâncias - Utilizando moda
- 'GarageType'    : 81  instâncias - Utilizando moda
- 'GarageYrBlt'   : 81  instâncias - Utilizando moda
- 'GarageFinish'  : 81  instâncias - Utilizando moda
- 'GarageQual'    : 81  instâncias - Utilizando moda
- 'GarageCond'    : 81  instâncias - Utilizando moda
- 'MasVnrType'    : 8   instâncias - Utilizando moda
- 'MasVnrArea'    : 8   instâncias - Utilizando média

**No conjunto de test:**
- 'LotFrontage'   : 227 instâncias - Utilizando média
- 'MSZoning'      : 4   instâncias - Utilizando moda
- 'Utilities'     : 2   instâncias - Utilizando moda
- 'Exterior1st'   : 1   instâncias - Utilizando moda
- 'Exterior2nd'   : 1   instâncias - Utilizando moda
- 'BsmtQual'      : 44  instâncias - Utilizando moda
- 'BsmtCond'      : 45  instâncias - Utilizando moda
- 'BsmtExposure'  : 44  instâncias - Utilizando moda
- 'BsmtFinType1'  : 42  instâncias - Utilizando moda
- 'BsmtFinSF1'    : 1   instâncias - Utilizando média
- 'BsmtFinType2'  : 42  instâncias - Utilizando moda
- 'BsmtFinSF2'    : 1   instâncias - Utilizando média
- 'BsmtUnfSF'     : 1   instâncias - Utilizando média
- 'TotalBsmtSF'   : 1   instâncias - Utilizando média
- 'BsmtFullBath'  : 2   instâncias - Utilizando moda
- 'BsmtHalfBath'  : 2   instâncias - Utilizando moda
- 'KitchenQual'   : 1   instâncias - Utilizando moda
- 'Functional'    : 2   instâncias - Utilizando moda
- 'GarageType'    : 76  instâncias - Utilizando moda
- 'GarageYrBlt'   : 78  instâncias - Utilizando moda
- 'GarageFinish'  : 78  instâncias - Utilizando moda
- 'GarageCars'    : 1   instâncias - Utilizando moda
- 'GarageArea'    : 1   instâncias - Utilizando moda
- 'GarageQual'    : 78  instâncias - Utilizando moda
- 'GarageCond'    : 78  instâncias - Utilizando moda
- 'MasVnrType'    : 16  instâncias - Utilizando moda
- 'MasVnrArea'    : 15  instâncias - Utilizando média
- 'SaleType'      : 1   instâncias - Utilizando moda

### Tratando Features nominais

Para atributos nominais foi aplicada a numerização, que consiste em substituir as classes de um atributo por números. Esse passo é necessário para que o treino pelos modelos de aprendizado de máquina seja possível.

Features numerizadas: 'MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood', 'Condition1','Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'

### Divisão dos dados

Para podermos analisar e melhorar os parâmetros utilizados no regressor, os dados serão separados em conjunto de treino e validação, sendo o conjunto de validação utilizado para fazermos um _tunning_ no modelo. Essa separação é necessária para verificarmos o tradeoff viés-variância antes de aplicarmos o modelo no conjunto de testes. Levando em consideração que temos em mãos um conjunto de dados relativamente grande, retiramos 20% do conjunto de treino para utilizarmos como validação de cada método.
