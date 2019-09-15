from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from datetime import datetime
import numpy as np

DTNow = datetime.now()
DateTime = DTNow.strftime("%d/%m/%Y %H:%M:%S") 

CaminhoTabela = r'~/IAFinancialmarket/B3SA3_1.2.1P.csv'
tabela = pd.read_csv(CaminhoTabela)

df = pd.DataFrame(tabela)

print(df)
#
X = df[['Fechamento','Abertura','Maxima','Minima','Chikou','Tekan','Kinju','Senkou A','Senkou B','Chikou0','Chikou1','Chikou2','T/K 0','T/K 1','T/K 2','Preco X T/K 0','Preco X T/K 1','Preco X T/K 2','Kumo 0','Kumo 1','Kumo 2']]
Y = df['Resultado']

ppn = MLPClassifier()

ppn.fit(X, Y)

y_pred = ppn.predict(X)

for i in y_pred
    print(y_pred)

#np.savetxt('~/IAFinancialmarket/test.csv',y_pred,delimiter=',')

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
    arquivo.write(str(' Accuracy: %.2f ' % accuracy_score(Y, y_pred)))
    arquivo.write('\n')
    arquivo.close()

print('Accuracy: %.2f' % accuracy_score(Y, y_pred))
