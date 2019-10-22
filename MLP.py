from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
# Producao
# from sklearn.externals import joblib

DTNow = datetime.now()
DateTime = DTNow.strftime("%d/%m/%Y %H:%M:%S") 

CaminhoTabela = r'~/Documentos/IAFinancialmarket/BTC_USD.csv'
CaminhoTabela_test = r'~/Documentos/IAFinancialmarket/BTC_USD_TEST.csv'

tabela = pd.read_csv(CaminhoTabela)
tabela_test = pd.read_csv(CaminhoTabela_test)

df = pd.DataFrame(tabela)
df_test = pd.DataFrame(tabela_test)

X = df[['Chikou0','Chikou1','Chikou2','T/K 0','T/K 1','T/K 2','Preco X T/K 0','Preco X T/K 1','Preco X T/K 2','Kumo 0','Kumo 1','Kumo 2']]
Y = df['Resultado']
X_test_manual = df_test[['Chikou0','Chikou1','Chikou2','T/K 0','T/K 1','T/K 2','Preco X T/K 0','Preco X T/K 1','Preco X T/K 2','Kumo 0','Kumo 1','Kumo 2']]
y_test_manual = df_test['Resultado']

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=0, stratify=Y)

ppn = MLPClassifier()

ppn.fit(X_train, y_train)

y_pred = ppn.predict(X_test_manual)

# Producao
# joblib.dump(ppn, 'mlpBTC_USD.pk1')

# A Seguir uma imagem com modelo que recebe duas entradas na camada um. 
# A camada dois, abriga duas funções de pertinência para cada entrada, mapeado para cada conjunto nebuloso. 
# Esses podem ser otimizados com algoritmos de treinamento ou de otimização. 
# A terceira camada, possui funções de ativação fixas. 
# A camada quatro, é responsável pela normalização dos valores processados anteriormente. 
# A quinta camada, é interligada com entrada para o processamento da função de ativação, que também pode passar por um processo de aprendizagem. 
# Na última camada, é feito o somatório de todas as saídas anteriores, assim apresentando a saída de ANFIS (Fonseca 2012). 


prediction = pd.DataFrame(y_pred, columns=['predictions']).to_csv('prediction.csv')

with open('Resultados.txt','a') as arquivo:
    arquivo.write('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    arquivo.write('\n')
    arquivo.write(str(DateTime))
    arquivo.write('\n')
    arquivo.write(str(CaminhoTabela))
    arquivo.write('\n')
    #arquivo.write(str(df.head(1)))
    arquivo.write(str(X.head(1)))
    arquivo.write('\n')
    arquivo.write(str(ppn))
    arquivo.write('\n')
    arquivo.write(str(train_test_split))
    arquivo.write('\n')
    # arquivo.write(str(CaminhoTabela_test))
    # arquivo.write('\n')
    # arquivo.write(str(X_test_manual.head(1)))
    # arquivo.write('\n')
    # arquivo.write(str(y_pred))
    # arquivo.write('\n')
    arquivo.write(str(' Accuracy: %.2f ' % accuracy_score(y_test_manual, y_pred)))
    arquivo.write('\n')
    arquivo.close()

print('Accuracy: %.2f' % accuracy_score(y_test_manual, y_pred))