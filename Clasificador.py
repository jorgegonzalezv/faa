from abc import ABCMeta,abstractmethod
import numpy as np
import matplotlib.pyplot as plt


TPR_=[]
FPR_=[]

""" Normal unidimensional

    x: valor 
    m: media 
    v: varianza

"""
def normal(x,m,v):
  return ((1/np.sqrt(np.pi*2*v)) * np.exp((-(x - m)**2) /(2*v)))

class Clasificador:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
  # de variables discretas
  @abstractmethod
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass  
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
    suma = 0
    for i, dato in enumerate(datos):
      if dato[-1] != pred[i]:
        suma = 1 + suma
        #print("\033[1;30;m"+str(dato))
    #  else:
        #print("\033[1;31;m"+str(dato))
  
    porcentajeAcierto = suma / len(pred)

    #print("\033[1;30;m"+str(pred))

    return porcentajeAcierto

  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):
    

    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test
    
    particiones = particionado.creaParticiones(dataset)

    errores = [] # errores cometidos con cada particion

    for particion in particiones:
      datosTrain = dataset.extraeDatos(particion.indicesTrain)
      datosTest = dataset.extraeDatos(particion.indicesTest)
      self.entrenamiento(datosTrain, dataset.nominalAtributos, dataset.diccionarios)
      resultados = self.clasifica(datosTest, dataset.nominalAtributos, dataset.diccionarios)
      errorParticion = self.error(datosTest, resultados)
      errores.append(errorParticion)

    return errores 
  
##############################################################################

class ClasificadorNaiveBayes(Clasificador):

  def __init__(self):
    self.frecuencias = [] # TODO posible cambio a diccionario (puede que mejor)
    self.probTrue = -1
    self.probFalse = -1

  # TODO: implementar
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
  # de variables discretas
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
      
      """ Objetivo:  Tenemos que crear las tablas de frecuencias de los atributos Nominales.
          Para los atributos cuantitativos no haria falta tabla,unicamente se recoge la media 
          y varianza (desviacion tipica) muestral. (suponiendo normalidad de variables continuas)

          Estructura de datos: Cada atributo nominal necesita dos tablas, las crearemos como diccionarios. Representan
          el atributo condicionado a clase = True o clase = False (1 o 0).


          TODO: correccion de Laplace
          - - - - - - - - - 
      """
      # Creacion de tablas en forma diccionarios
      # ej: balloons.data
      #     atributo(variable): color
      #       diccionario {
      #           Yellow -> [a, b]
      #           Purple -> [c ,d]
      #           }
      #
      #       Color / Clase | False | True
      #       -----------------------------  
      #              Yellow |   a  |  b
      #              Purple |   c  |  d

      # reset variables (en caso de mas de una llamada )
      self.frecuencias = [] 
      self.probTrue = -1
      self.probFalse = -1
      

      # recorrido todos los diccionarios que codifican las
      # variables discretas. ver Datos.py
      for i, dic in enumerate(diccionario):
        dicAtribributo = {} 

        # nuevo diccionario atributo discreto
        #
        #       Color / Clase | False | True
        #       -----------------------------  
        #              Yellow |   0  |  0
        #              Purple |   0  |  0
        #
        if atributosDiscretos[i] == True:
          for valor in dic.keys():
            dicAtribributo[valor] = [0, 0]

        #       Atributo continuo| False | True
        #       ------------------------------------
        #          media         |   0   |  0
        #          varianza      |   0   |  0
        #
        #

        else:
          dicAtribributo['media'] = [0, 0]
          dicAtribributo['varianza'] = [0, 0]

        # añadimos a la lista de tablas (diccionarios)
        self.frecuencias.append(dicAtribributo)

      #recorrido de filas
      for dato in datostrain:
        veroTrue = diccionario[-1].get(dato[-1]) # 0 si False, 1 si True 

        # recorrido de columna
        for i, col in enumerate(dato):
          # incrementamos en uno el valor correspondiente
          # [atributo][valor][true/false]
          if atributosDiscretos[i] == True:
            self.frecuencias[i][col][veroTrue] = 1 + self.frecuencias[i][col][veroTrue]

          # caso continuo calculamos suma de todos lo valores
          else:
            self.frecuencias[i]['media'][veroTrue] =  self.frecuencias[i]['media'][veroTrue] + float(col)


      falso = list(diccionario[-1].keys())[0]
      verdad = list(diccionario[-1].keys())[1]

      totalFalse = self.frecuencias[-1][falso][0]
      totalTrue = self.frecuencias[-1][verdad][1]
      total = datostrain.shape[0]

      # obtencion medias muestrales para atributos continuos
      for i in range(len(self.frecuencias)):
          if atributosDiscretos[i] == False:
            self.frecuencias[i]['media'][0] = self.frecuencias[i]['media'][0] / totalFalse
            self.frecuencias[i]['media'][1] = self.frecuencias[i]['media'][1] / totalTrue

      # obtencion varianzas muestrales para atributos continuos
      for dato in datostrain:
        for i, col in enumerate(dato):
          if atributosDiscretos[i] == False:
            self.frecuencias[i]['varianza'][0] = ((float(col) - (self.frecuencias[i]['media'][0]))**2/ totalFalse) + self.frecuencias[i]['varianza'][0]
            self.frecuencias[i]['varianza'][1] = ((float(col) - (self.frecuencias[i]['media'][1]))**2/ totalTrue) + self.frecuencias[i]['varianza'][1]

      # el atributo clase siempre es nominal
      self.probFalse = totalFalse / total
      self.probTrue = totalTrue / total

      # -> debbuging
      #for dic in self.frecuencias:
      #  print("- - - - - - - - - - - - - - - - - - - - - - -")
      #  print(dic)

  def clasifica(self,datostest,atributosDiscretos,diccionario):

      """ Objetivo:  Debemos leer la tablas correspondientes y obtener las verosimilitudes
          a partir de las frecuencias.

          P(clase = True | datos) = P(datos | clase = true) * P(clase = true) / P(datos) donde datos = (atributo1 = x1, ..., atributoN = xN)

          Naive Bayes supone independencia de los atributos dado la clase:
          
            P(datos | true) = P(atributo1 = x1 | true) * .... * P(atributoN = xN| true)

            Por ejemplo, el dato i-esimo, correspondiente a una variable aleatoria que toma ciertos valores
            discretos {a,b,c, ... }.

            P(atributo1 = x1 | true) = # filas con atributo1 = x1 y clase = true / # total de filas con clase = true
          - - - - - - - - - 
      """

      results = []

      falso = list(diccionario[-1].keys())[0]
      verdad = list(diccionario[-1].keys())[1]

      totalFalse = self.frecuencias[-1][falso][0]
      totalTrue = self.frecuencias[-1][verdad][1]

      for dato in datostest:

        # inicio productorio
        freqColTrue = 1
        freqColFalse = 1
        for i, col in enumerate(dato[:-1]): # menos la ultima casilla que es la clase

          # caso atributo discreto
          if atributosDiscretos[i] == True:
            freqColTrue = (self.frecuencias[i][col][1]/totalTrue) * freqColTrue
            freqColFalse = (self.frecuencias[i][col][0]/totalFalse) * freqColFalse

          # caso atributo continuo
          else:
            freqColTrue = normal(float(col),self.frecuencias[i]['media'][1],self.frecuencias[i]['varianza'][1]) * freqColTrue
            freqColFalse = normal(float(col),self.frecuencias[i]['media'][0],self.frecuencias[i]['varianza'][0]) * freqColFalse

        freqColTrue = freqColTrue * self.probTrue
        freqColFalse = freqColFalse * self.probFalse

        if freqColTrue < freqColFalse:
          results.append(falso)
        else:
          results.append(verdad)

      self.curvaROC(datostest, results)

      TPR_=[]
      FPR_=[]

      return np.array(results)

  def curvaROC(self,datostest,results):  

    """ Objetivo:  Calcular la razon entre verdaderoos positivos y falsos positivos con el fin de ver la sensibilidad y 1-espcifidad que obtenemos 
                    clasificando una muestra de los datos.
    """

    valoresReales = []

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    TPR = 0.0
    FNR = 0.0
    FPR = 0.0
    TNR = 0.0

    #Obtenemos en un array el valor real de la clase de las instancias que hemos clasificado
    for dato in datostest:
      valoresReales.append(dato[-1])

    for i in range(len(results)):
      if results[i] == valoresReales[0]:
        if (results[i] == valoresReales[i]):
          TP = TP+1 #TRUE POSITIVE
        else:
          FP = FP+1 #FALSE POSITIVE
      else:
        if (results[i] == valoresReales[i]):
          TN = TN+1 #TRUE NEGATIVE
        else:
          FN = FN+1 #FALSE NEGATIVE

    if (TP+FN != 0):
      TPR = TP/(TP+FN) #Se dibuja en el eje Y
      TPR_.append(TPR)
      FNR = FN/(TP+FN)
    else:
      TPR_.append(TPR)

    if (FP+TN != 0):
      FPR = FP/(FP+TN) #Se dibuja en el eje X
      FPR_.append(FPR)
      TNR = TN/(FP+TN)
    else:
      FPR_.append(FPR)

  def plotROC(self):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    curve2 = ax.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR or (1 - specificity)')
    plt.ylabel('TPR or sensitivity')
    plt.title('ROC curve')

    plt.scatter(FPR_, TPR_)
    plt.show()
    #plt.savefig("CurvaROC.png")






    
    





  