from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

tabelaOR = pd.read_csv(r'')

df = pd.DataFrame(tabelaOR)
print(df)

X = df[['x1','x2']]
Y = df['y']


ppn = Perceptron(n_iter=10, eta0=0.2, random_state=0)

ppn.fit(X, Y)

y_pred = ppn.predict(X)

print(y_pred)

print('Accuracy: %.2f' % accuracy_score(Y, y_pred))