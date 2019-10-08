from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
# Producao
from sklearn.externals import joblib

DTNow = datetime.now()
DateTime = DTNow.strftime("%d/%m/%Y %H:%M:%S") 

CaminhoTabela = r'~/IAFinancialmarket/BTC_USD.csv'
tabela = pd.read_csv(CaminhoTabela)

df = pd.DataFrame(tabela)

print(df)

X = df[['Fechamento','Abertura','Maxima','Minima','Chikou','Tekan','Kinju','Senkou A','Senkou B','Chikou0','Chikou1','Chikou2','T/K 0','T/K 1','T/K 2','Preco X T/K 0','Preco X T/K 1','Preco X T/K 2','Kumo 0','Kumo 1','Kumo 2']]
Y = df['Resultado']

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=42)

ppn = MLPClassifier()

ppn.fit(X_train, y_train)

#X_teste_manual =

y_pred = ppn.predict(X_test)

# Producao
joblib.dump(ppn, 'mlpBTC_USD.pk1')

prediction = pd.DataFrame(y_pred, columns=['predictions']).to_csv('prediction1.csv')

with open('Resultados.txt','a') as arquivo:
    arquivo.write('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    arquivo.write('\n')
    arquivo.write(str(DateTime))
    arquivo.write('\n')
    arquivo.write(str(CaminhoTabela))
    arquivo.write('\n')
    arquivo.write(str(df.head(1)))
    arquivo.write('\n')
    arquivo.write(str(ppn))
    arquivo.write('\n')
    arquivo.write(str(y_pred))
    arquivo.write('\n')
    arquivo.write(str(' Accuracy: %.2f ' % accuracy_score(y_train, y_pred)))
    arquivo.write('\n')
    arquivo.close()

print('Accuracy: %.2f' % accuracy_score(y_train, y_pred))


# escrever na base de dados