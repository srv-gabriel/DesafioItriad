# DesafioItriad

O objetivo do desafio é desenvolver um modelo capaz de predizer o valor de um determinado imóvel baseado em suas características. O conjunto de dados inicial apresenta 80 características mais uma coluna com o valor de venda do imóvel, que é a coluna-alvo.

## Como Utilizar
Basta clonar ou fazer o download do repositório, abrir o Jupyter Notebook e executar as células, caso esteja em um Linux ou MacOS. Os dados são lidos do diretório dataset e o resultado final é salvo em formato CSV também no repositório dataset. Caso o sistema operacional for um windows é necessário informar o path de onde o arquivo será lido e o path onde o arquivo será salvo, no Jupyter Notebook.

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

## Regressores

### Redução de dimensionalidade e normalização

Nesta seção é aplicado o algoritmo de redução de dimensionalidade PCA. Ao utilizarmos algoritmos mais simples, como Random Forest e SVM devemos ter cuidado com a quantidade de features, e.g. dimensão, dos nossos dados, pois algumas características podem inserir ruído e acabar por "confundir" o modelo.

O PCA é um algoritmo não-supervisionado que analisa a variância entre cada feature e gera uma transformação linear de forma a maximizar essa variância. Com a variância maximizada, características que não acrescentam ao modelo, ou que prejudicam, são retiradas.

Para selecionar a quantidade de componentes do PCA que utilizaremos, passamos o valor .95, que indica que queremos o número de features que representam 95% da variância da base de dados. Dessa forma, são retornadas 54 features, que são utilizadas para o Rnadom Forest e para o SVM. Como as features geradas pelo PCA são uma combinação de outras características, elas não apresentam um real significa como a features originais.

Para a Rede Neural, devido ao método ser mais robusto e complexo, utilizaremos o conjunto de dados original, com 74 features.

Também é aplicada normalização dos dados do conjunto. a normalização é feita de tal forma que a distribuição de todas as características estejam entre 0 e 1. Como o conjunto apresenta uma variação bem grande na distribuição dos dados, a normalização acaba por facilitar a identificação de determinados padrões pelo modelo. A normalização é feita através da subtração da média e divisão pelo desvio padrão em cada feature separadamente

### Random Forest Regressor

O Random Forest Regressor é um algoritmo de ensemble que utiliza o método de Bagging. Funciona da seguinte forma: o conjunto de dados é separado em vários subconjuntos menores, e cada conjunto desse é independente, ou seja, não possuem dados sobressalentes. Então são contruídas várias árvores profundas e treinadas com diferentes subconjuntos de forma que cada árvore extraia diferentes conceitos de cada subconjunto. Então as árvores são agregadas e a média do resultado de todas as árvores é a saída final do Random Forest

Em geral, um conjunto de regressores fracos apresentam melhor resultado que um único regressor forte.

Foi utilizado o GridSearch Cross-Validation com 10 partições para identificar o conjunto de hiperparâmetros que tem a melhor performance, utilizando RMSE como métrica.

**Os hiperparâmetros definidos através do GridSeatch foram:**
 - max_depth    : 6;
 - n_estimators : 100.

### SVM

O SVM, usualmente muito utilizado para classificação, também pode ser utilizado para regressão. Os princípios do algorítmo continuam o mesmo: definir um hiperplano e maximizar a margem entre os dados. O algorítmo apresenta certa aceitação de erros, o que pode prejudicar sua performance em problemas de regressão, já que a saída é em intervalo contínuo e apresenta infinitas possibilidades. O SVM é um modelo muito robusto e na "era" pré Deep Learning era tão usado quanto as redes neurais atualmente, devido a sua capacidade de aprender diversas funções complexas.

Fazemos aqui novamente o uso do GridSearch para encontrarmos os melhores parâmetros possíveis dentro de um espaço de busca pré-definido.

**Os hiperparâmetros definidos através do GridSeatch foram:**
 - C      : 200
 - degree : 2
 - gamma  : 0.25
 - kernel : sigmoid

### Rede Neural

Um dos maiores passos da computação nesse década, o Deep Learning apresenta robustez maior que os algorítmos de Aprendizagem clássicos apresentados anteriormente. Uma Rede Neural consiste basicamente da tentativa de encontrar o mínimo global do gradiente de uma função de perda. Ela é composta por um número $N$ de camadas e cada camada $N_i$ possui uma quantidade $X_i$ de neurônios e seus respectivos pesos. A rede neural recebe um vetor de features e o resultado baseado na função de ativação que cada camada recebeu. Após isso, é computado o erro utilizando uma função de perda e os pessoas da rede neural são atualizados.

Devido a complexidade e robustez do método, foi utilizado o banco de dados original, sem aplicação do PCA ou Normalização das características.

**O modelo criado para esse problema utiliza:**

-   4 camadas:
    -   primeira camada: 30 neurônios; função de ativação: ReLU; Tamanho de entrada: 74 features;
    -   segunda camada: 15 neurônios; função de ativação: ReLU;
    -   terceira camada: 20 neurônios; função de ativação: ReLU;
    -   camada de saída: 1 neurônios; função de ativação: Linear.
-   função de perda: Mean Squared Error;
-   otimizador: adam;
-   métricas utilizadas: Mean Absolute Error, Mean Squared Logarithmic Error;
-   Épocas: 1000;
-   Batch Size: 10;
-   Conjunto de Validação: 20% do conjunto original.
