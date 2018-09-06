import numpy as np


class Perceptron():

    def __init__(self, data, function, hidden_neurons=5,neurons=3, proportion=0.8, eta=0.1, epochs=800):
        self.data = data  # Base de dados
        self.functionName = function[0]  # Recebe o nome da função de ativação
        self.function = function[1]  # Recebe a função de ativação (y)
        self.functionDerivative = function[2]  # Recebe a derivada da função (y')
        self.proportion = proportion  # treina com 80% dos dados e testa com 20% dos dados
        self.eta = eta  # Taxa de aprendizagem
        self.epochs = epochs  # Número máximo de épocas
        self.neurons = neurons
        self.hidden_neurons = hidden_neurons
        self.bias = -1.0  # x0
        self.theta = -1.0  # w0
        self.h0 = -1.0 #h0
        self.data = self.insertBias()  # insere valor do x0 na base de dados
        self.wi = self.initW(0)  # incializa w da camada oculta (aleatório com theta w0)
        self.wj = self.initW(1)  # incializa w da camada de saida (aleatório com theta w0)

    # Retorna a ultima coluna de certo x com o valor de classe desejado
    # Filtra a classe pelo index do neuronio
    def desired(self, x):
        d = []
        label = x[x.size - 1]
        if self.functionName == 'Tangente hiperbólica':
            for index in range(self.neurons):
                if label == index:
                    d.append(-1)
                else:
                    d.append(1)
        else:
            for index in range(self.neurons):
                if label == index:
                    d.append(1)
                else:
                    d.append(0)
        return d

    # Insere o valor do bias(x0) na base de dados recebida
    def insertBias(self):
        d = []
        for i in range(len(self.data)):
            d.append(np.insert(self.data[i], 0, self.bias))  # insere o valor de x0 para todos os padrões
        d = np.asarray(d)
        return d

    # Inicializa a matrix w com mesmo numero de colunas da base e com c linhas(número de neuronios)
    def initW(self,layer_index):
        if layer_index == 0:
            matrix = np.random.rand(self.hidden_neurons, (self.data.shape[1] - 1))
            matrix[:, 0] = self.theta
        else:
            matrix = np.random.rand(self.neurons, (self.data.shape[1] - 1))
            matrix[:, 0] = self.theta
        return matrix

    # Imprime informações gerais sobre o modelo da rede
    def printInfo(self):
        print("Informações: \n")
        print("Dados:", self.data)
        # print("Proporção de treinamento/testes:", self.proportion)
        # print("Taxa de aprendizagem:", self.eta)
        # print("Número de épocas: ", self.epochs)
        print("Vetor wi inicial: \n", self.wi)
        print("Vetor wj inicial: \n", self.wj)
        # print("Função de ativação:", self.functionName)

    # Calcula o produto interno w[i]T.x
    # Onde, i é o index do neurônio
    def dotProduct(self, pattern, weights):
        x = pattern[0:-1]
        u = []
        for index in range(len(weights)):
            u.append(np.dot(weights[index], x))
        return u
    
    # TODO ajustar calculo do y
    # Retorna uma lista com as saidas dos neuronios
    def y(self, hidden_output):
        print('h: ', hidden_output)
        print('wj: ', self.wj)
        u = self.dotProduct(hidden_output, self.wj)
        y = []
        for index in range(self.neurons):
            y.append(self.function(u[index]))
        return y

    def filter(self, e):
        filtered_e = np.zeros(self.neurons)
        maxIndex = np.argmax(np.absolute(e))
        for i in range(len(e)):
            if i > maxIndex:
                filtered_e[i] = np.absolute(e[i])
            else:
                filtered_e[i] = 0
        return filtered_e
    #TODO: consertar regra de aprendizagem
    # REGRA DE APRENDIZAGEM / AJUSTE DO VETOR W
    # w(t+1)=w(t) + (taxa_aprendizagem * erro_iteração)*x(t)
    def adjust_wj(self, pattern, error, y):
        x = pattern[0:-1]
        sum = 0
        for j in range(self.neurons):
            self.wj[j] = self.wj[j] + (self.eta) * (error[j]) * (self.functionDerivative(y[j])) * x

        for i in range(self.hidden_neurons):
            for j in range(self.neurons):
                sum = sum + (self.wj[j] * error[j] * self.functionDerivative(y[j]))
            self.wi[i] = sum

    # Retorna uma lista de erros de cada neuronio
    def error(self, x, y):
        e = []
        d = self.desired(x)
        for index in range(self.neurons):
            e.append(d[index] - y[index])
        return np.array(e)

    # Saída da camada oculta
    def h(self, x):
        u = self.dotProduct(x,self.wi)
        print('ui ', u)
        h = [self.h0]
        for index in range(self.hidden_neurons):
            h.append(self.function(u[index]))
        return np.array(h)

    def training(self):
        data = self.data[0: int(len(self.data) * self.proportion)]  # utiliza apenas a proporção certa dos dados
        i = 1
        while i < self.epochs:
            np.random.shuffle(data)  # shuffle entre épocas
            for x in data:
                h = self.h(x)
                y = self.y(h)
                error = self.error(x, y)
                # todo metodo de ajustes
                self.adjust_wj(x, error, y)
                self.adjust_wi(x, error, y)
            i += 1
        return self.wj

    # TESTES
    def test(self):
        data = self.data[int(len(self.data) * self.proportion):]  # utiliza apenas a proporção certa dos dados
        acc = []
        tolerance = 0.2
        for x in data:
            if self.functionName != 'Step function':
                e = sum(self.filter(self.error(x)))
                if e <= tolerance:
                    acc.append(1)
                else:
                    acc.append(0)
            else:
                e = sum(self.error(x))
                if e == 0:
                    acc.append(1)
                else:
                    acc.append(0)

        return sum(acc) / len(data)

    # REALIZAÇÃO
    # Faz uma realização completa, que consiste em:
    #   - Treino
    #   - Testes
    def execution(self, times):
        acc_tx = []  # lista que salva a taxa de acerto de cada realização
        q = 0
        dataToPlot = self.data
        wToPlot = self.wj
        print("### PERCEPTRON SIMPLES ###")
        print("PARÂMETROS: ")
        self.printInfo()
        print("Total de realizações: ", times, "\n")

        for i in range(1, times + 1):
            print("### REALIZAÇÃO ", i, "###")
            np.random.shuffle(self.data)  # shuffle entre realizações
            self.wj = self.initW()  # reseta o vetor de pesos entre realizações
            print("### FASE DE TREINAMENTO ###")
            w = self.training()
            print("Vetor W final: \n", w)
            print("### FASE DE TESTES ###")
            tx = self.test()
            print("Taxa de acerto: ", tx, "\n")
            acc_tx.append(tx)
            if tx >= q:
                q = tx
                dataToPlot = self.data
                wToPlot = self.wj

        accuracy = (sum(acc_tx) / times)  # acurácia entre [0,1]
        # Cálculo do desvio padrão
        dp = self.standardDeviation(accuracy, acc_tx)
        accuracy *= 100  # acurácia em porcentagem
        print("DESVIO PADRÃO: ", dp)
        print("ACURÁCIA: ", accuracy)
        print("### FIM PERCEPTRON ###")
        return [q, dataToPlot, wToPlot]

    def standardDeviation(self, mean, list):
        aux = []
        for i in list:
            aux.append((i - mean) ** 2)
        d = np.sqrt(sum(aux) / len(list))
        return d
