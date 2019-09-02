from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

tabelaXOR = pd.read_csv(r'xor.csv')

df = load_wine(return_X_y=True)

X = df[0][1:,2:]
Y = df[1][0:-1]


ppn = Perceptron(n_iter=10, eta0=0.2, random_state=0)

ppn.fit(X, Y)

y_pred = ppn.predict(X)

print(y_pred)

print('Accuracy: %.2f' % accuracy_score(Y, y_pred))