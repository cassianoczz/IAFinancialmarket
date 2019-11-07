from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

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
# A camada dois, abriga duas funcoes de pertinencia para cada entrada, mapeado para cada conjunto nebuloso. 
# Esses podem ser otimizados com algoritmos de treinamento ou de otimizacao. 
# A terceira camada, possui funcoes de ativacao fixas. 
# A camada quatro, e responsavel pela normalizacao dos valores processados anteriormente. 
# A quinta camada, e interligada com entrada para o processamento da funcao de ativacao, que tambem pode passar por um processo de aprendizagem. 
# Na ultima camada, e feito o somatorio de todas as saidas anteriores, assim apresentando a saida de ANFIS (Fonseca 2012). 


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

# New Antecedent/Consequent objects hold universe variables and membership
# functions

#RNA = ctrl.Antecedent(np.arange(0, 3, 1), 'RNA')
PrecoxNuvem = ctrl.Antecedent(np.arange(0, 11, 1), 'PrecoxNuvem')
SenkouBfuturaReta = ctrl.Antecedent(np.arange(0, 11, 1), 'SenkouBfuturaReta')
KumoGrossa = ctrl.Antecedent(np.arange(0, 11, 1), 'KumoGrossa')
PrecoLongeKinju = ctrl.Antecedent(np.arange(0, 11, 1), 'PrecoLongeKinju')
Pullback = ctrl.Antecedent(np.arange(0, 11, 1), 'Pullback')
KinjuReta = ctrl.Antecedent(np.arange(0, 11, 1), 'KinjuReta')
Lateralizacao = ctrl.Antecedent(np.arange(0, 11, 1), 'Lateralizacao')

# Auto-membership function population is possible with .automf(3, 5, or 7)
PrecoxNuvem.automf(3)
SenkouBfuturaReta.automf(3)
KumoGrossa.automf(3)
PrecoLongeKinju.automf(3)
Pullback.automf(3)
KinjuReta.automf(3)
Lateralizacao.automf(3)

#Graphics
#RNA.view()
#ANOTACAO: PASSAR OS ARGUMETNOS PARA A FUNCAO
def Comprado():

    comprado = ctrl.Consequent(np.arange(0, 101, 1), 'comprado')

    comprado.automf(3)

    # comprado['fraco'] = fuzz.trimf(comprado.universe, [0, 0, 40])
    # comprado['medio'] = fuzz.trimf(comprado.universe, [0, 40, 60])
    # comprado['forte'] = fuzz.trimf(comprado.universe, [40, 60, 100])

    regra1 = ctrl.Rule(
        #RNA['comprado'] & 
        PrecoxNuvem['average'] & SenkouBfuturaReta['poor'] & KumoGrossa['poor'] & (PrecoLongeKinju['average'] | (KinjuReta['good'] & PrecoLongeKinju['good'] | Pullback['good'])), comprado['good'])
    regra2 = ctrl.Rule(
        #RNA['comprado'] & 
        (PrecoxNuvem['poor'] | SenkouBfuturaReta['average']) & KumoGrossa['average'] & (PrecoLongeKinju['average'] | Pullback['average']) & Lateralizacao['poor'], comprado['average'])
    regra3 = ctrl.Rule(
        #RNA['comprado'] & 
        SenkouBfuturaReta['good'] & KumoGrossa['good'] & Lateralizacao['good'], comprado['poor'])
    
    #regra1.view()

    comprado_ctrl = ctrl.ControlSystem([regra1, regra2, regra3])
    comprado_ctrl_Sim = ctrl.ControlSystemSimulation(comprado_ctrl)

    # comprado_ctrl_Sim.input['RNA'] = y_pred
    comprado_ctrl_Sim.input['PrecoxNuvem'] = 2
    comprado_ctrl_Sim.input['SenkouBfuturaReta'] = 9
    comprado_ctrl_Sim.input['KumoGrossa'] = 2
    comprado_ctrl_Sim.input['PrecoLongeKinju'] = 0
    comprado_ctrl_Sim.input['Pullback'] = 2
    comprado_ctrl_Sim.input['KinjuReta'] = 2
    comprado_ctrl_Sim.input['Lateralizacao'] = 2

    # Crunch the numbers
    comprado_ctrl_Sim.compute()

    return comprado_ctrl_Sim.output['comprado']

def Aguardar():
    aguardar = ctrl.Consequent(np.arange(0, 100, 1), 'aguardar')
    
    aguardar.automf(3)

    #aguardar['fraco'] = fuzz.trimf(aguardar.universe, [0, 0, 40])
    #aguardar['medio'] = fuzz.trimf(aguardar.universe, [0, 40, 60])
    #aguardar['forte'] = fuzz.trimf(aguardar.universe, [40, 60, 100])
    
    regra1 = ctrl.Rule(SenkouBfuturaReta['good'] & Lateralizacao['good'] & Pullback['poor'] & KinjuReta['good'], aguardar['good'])
    regra2 = ctrl.Rule(PrecoxNuvem['poor'] & SenkouBfuturaReta['average'] & Pullback['poor'] & Lateralizacao['average'], aguardar['average'])
    regra3 = ctrl.Rule(PrecoxNuvem['average'] & SenkouBfuturaReta['poor'] | KinjuReta['poor'] | Lateralizacao['poor'], aguardar['poor'])

    #regra1.view()

    aguardar_ctrl = ctrl.ControlSystem([regra1, regra2, regra3])
    aguardar_ctrl_Sim = ctrl.ControlSystemSimulation(aguardar_ctrl)

    aguardar_ctrl_Sim.input['PrecoxNuvem'] = 2
    aguardar_ctrl_Sim.input['SenkouBfuturaReta'] = 9
    #aguardar_ctrl_Sim.input['KumoGrossa'] = 2
    #aguardar_ctrl_Sim.input['PrecoLongeKinju'] = 0
    aguardar_ctrl_Sim.input['Pullback'] = 2
    aguardar_ctrl_Sim.input['KinjuReta'] = 2
    aguardar_ctrl_Sim.input['Lateralizacao'] = 2

    # Crunch the numbers
    aguardar_ctrl_Sim.compute()

    return aguardar_ctrl_Sim.output['aguardar']

def Vendido():
    vendido = ctrl.Consequent(np.arange(0, 100, 1), 'vendido')
    
    vendido.automf(3)

    # vendido['fraco'] = fuzz.trimf(vendido.universe, [0, 0, 40])
    # vendido['medio'] = fuzz.trimf(vendido.universe, [0, 40, 60])
    # vendido['forte'] = fuzz.trimf(vendido.universe, [40, 60, 100])
    
    regra1 = ctrl.Rule(PrecoxNuvem['average'] & SenkouBfuturaReta['poor'] & KumoGrossa['poor'] & (PrecoLongeKinju['average'] | (KinjuReta['good'] & PrecoLongeKinju['good'] | Pullback['good'])), vendido['good'])
    regra2 = ctrl.Rule((PrecoxNuvem['poor'] | SenkouBfuturaReta['average']) & KumoGrossa['average'] & (PrecoLongeKinju['average'] | Pullback['average']) & Lateralizacao['average'], vendido['average'])
    regra3 = ctrl.Rule(SenkouBfuturaReta['good'] & KumoGrossa['good'] & Lateralizacao['good'], vendido['poor'])

    regra1.view()

    vendido_ctrl = ctrl.ControlSystem([regra1, regra2, regra3])
    vendido_ctrl_Sim = ctrl.ControlSystemSimulation(vendido_ctrl)

    #comprado_ctrl_Sim.inputs(['PrecoxNuvem'] = 2,['SenkouBfuturaReta'] = 9)

    comprado_ctrl_Sim.input['PrecoxNuvem'] = 1
    comprado_ctrl_Sim.input['SenkouBfuturaReta'] = 9
    comprado_ctrl_Sim.input['KumoGrossa'] = 1
    comprado_ctrl_Sim.input['PrecoLongeKinju'] = 0
    comprado_ctrl_Sim.input['Pullback'] = 1
    comprado_ctrl_Sim.input['KinjuReta'] = 1
    comprado_ctrl_Sim.input['Lateralizacao'] = 1

    # Crunch the numbers
    vendido_ctrl_Sim.compute()

    return vendido_ctrl_Sim.output['vendido']


# if RNA = comprado 
print(Comprado())
print(Aguardar())
print(Comprado())
# return medium fraco
# else fuzzy.aguardar
# return medium fraco
# else fuzzy.vendido

# #tip.view(sim=tipping)
# #comprado().view()

# if RNA = aguardar 
# fuzzy.aguardar()
# return medium fraco
# else fuzzy.comprado
# return medium fraco
# else fuzzy.vendido

# if RNA = vendido 
# fuzzy.vendido()
# return medium fraco
# else fuzzy.comprado
# return medium fraco
# else fuzzy.aguardar