from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

tabela = pd.read_csv(r'~/IAFinancialmarket/BTC_USD_1.2.1.csv')

df = pd.DataFrame(tabela)
print(df)

X = df[['Date','Fechamento','Abertura','Maxima','Minima'
#,'Chikou','Tekan','Kinju','Senkou A','Senkou B','Chikou0','Chikou1','Chikou2','T/K 0','T/K 1','T/K 2','Preco X T/K 0','Preco X T/K 1','Preco X T/K 2','Kumo 0','Kumo 1','Kumo 2'
]]
Y = df['Resultado']

ppn = MLPClassifier()

ppn.fit(X, Y)

y_pred = ppn.predict(X)

print(y_pred)

print('Accuracy: %.2f' % accuracy_score(Y, y_pred))