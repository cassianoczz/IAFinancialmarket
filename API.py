#https://www.linkedin.com/pulse/criando-um-modelo-preditivo-e-colocando-em-produ%C3%A7%C3%A3o-maciel-guimar%C3%A3es
from flask import Flask, jsonify, request

# [1] importo o deserializador
from sklearn.externals import joblib 

# [2] Carrego a classe de predicao do diretorio local
ppn = joblib.load('mlpBTC_USD.pk1')

app = Flask(__name__)

@app.route('/BTC_USD')
def BTC_USD():
    # [3] Recupero as informacoes de preco
	Fechamento = float(request.args.get('Fechamento')) 
	Abertura = float(request.args.get('Abertura'))
	Maxima = float(request.args.get('Maxima')) 
	Minima = float(request.args.get('Minima'))
	# [3] Recupero as informacoes de Ichimoku
	Chikou = float(request.args.get('Chikou'))
	Tekan = float(request.args.get('Tekan'))
	Kinju = float(request.args.get('Kinju'))
	Senkou_A = float(request.args.get('Senkou_A'))
	Senkou_B = float(request.args.get('Senkou_B'))
	# [3] Daqui para baixo eh so conta
	Chikou0 = float(request.args.get('Chikou0'))
	Chikou1 = float(request.args.get('Chikou1'))
	Chikou2 = float(request.args.get('Chikou2'))
	TK_0 = float(request.args.get('T/K_0'))
	TK_1 = float(request.args.get('TK_1'))
	TK_2 = float(request.args.get('TK_2'))
	Preco_X_TK_0 = float(request.args.get('Preco_X_TK_0'))
	Preco_X_TK_1 = float(request.args.get('Preco_X_TK_1'))
	Preco_X_TK_2 = float(request.args.get('Preco_X_TK_2'))
	Kumo_0 = float(request.args.get('Kumo_0'))
	Kumo_1 = float(request.args.get('Kumo_1'))
	Kumo_2 = float(request.args.get('Kumo_2'))
	
	event = [Fechamento, Abertura, Maxima, Minima, Chikou, Tekan, Kinju, Senkou_A, Senkou_B, Chikou0, Chikou1, Chikou2, TK_0, TK_1, TK_2,Preco_X_TK_0, Preco_X_TK_1, Preco_X_TK_2, Kumo_0, Kumo_1, Kumo_2]
	target_names = ['1', '2', '0']

	result = {}

    # [4] Realiza predicao com base no evento
	prediction = ppn.predict([event])[0]

    # [5] Realizar probabilidades individuais das tres classes
	probas = zip(target_names, ppn.predict_proba([event])[0])

    # [6] Recupera o nome real da classe
	result['prediction'] = target_names[prediction]
	result['probas'] = probas

	return jsonify(result), 200

app.run()
