from flask import Flask, jsonify, request

# [1] importo o deserializador
from sklearn.externals import joblib 

# [2] Carrego a classe de predição do diretório local
ppn = joblib.load('mlpBTC_USD.pk1')

app = Flask(__name__)

@app.route('/BTC_USD')
def BTC_USD():

    # [3] Recupero as informações de uma Flor
	sepal_length = float(request.args.get('sepal_length'))
	sepal_width = float(request.args.get('sepal_width'))
	petal_length = float(request.args.get('petal_length'))
	petal_width = float(request.args.get('petal_width'))

	event = [sepal_width, sepal_width, petal_length, petal_width]
	target_names = ['Setosa', 'Versicolor', 'Virginica']

	result = {}

    # [4] Realiza predição com base no evento
	prediction = clf.predict([event])[0]

    # [5] Realizar probabilidades individuais das três classes
	probas = zip(target_names, clf.predict_proba([event])[0])

    # [6] Recupera o nome real da classe
	result['prediction'] = target_names[prediction]
	result['probas'] = probas

	return jsonify(result), 200

app.run()


https://www.linkedin.com/pulse/criando-um-modelo-preditivo-e-colocando-em-produ%C3%A7%C3%A3o-maciel-guimar%C3%A3es