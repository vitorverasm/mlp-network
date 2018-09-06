from src.perceptron import Perceptron
import utils.functions as functions
from utils.dataManipulation import DataManipulation
import utils.plot as plotData
# CAMINHOS DAS BASES DE DADOS
iris_path = './samples/iris.data'
art_path = './samples/artificial.data'

# PARAMETROS
problem = 'Iris'
realizacoes = 1
fn = functions.step_fn()
todosAtributos = False

if problem == 'Iris':
    dm = DataManipulation(iris_path, 0)
    if todosAtributos:
        data = dm.getData()  # Base iris com apenas todos atributos
    else:
        data = [[p[2:]] for p in dm.getData()]  # Base iris com apenas 2 atributos
else:
    dm = DataManipulation(art_path, 1)
    data = dm.getData()  # Base artificial

p = Perceptron(data, fn)
p.printInfo()
h=p.h([-1,-10,0.875,2])
print('h ',h)
print('y ',p.y(h))