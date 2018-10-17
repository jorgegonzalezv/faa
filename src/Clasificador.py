from abc import ABCMeta,abstractmethod


class Clasificador(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
  # de variables discretas
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
    pass
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):
       
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test
    pass
       
  
##############################################################################

class ClasificadorNaiveBayes(Clasificador):

 
  def __init__(self):
    self.verosimilitudes = [[],[]]

  # TODO: implementar
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
  # de variables discretas
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
      # calculo de verosimilitudes
      # conteo para variables discretas/ media y varianza suponiendo normalidad en variables continuas
      # TODO posible discretizacion de las variables continuas
      # vero false, vero true
      countTrue = 0
      total = 0
      
      #copiamos array!!
      self.verosimilitudes[0] = np.array(diccionario)
      for dic in diccionario:
        verosimilitudes[0].append(dict(zip(dic.keys(),[0]*len(dic))))
        verosimilitudes[1].append(dict(zip(dic.keys(),[0]*len(dic))))

      for dato in datostrain:
        # vemos la clase de la entrada (true o false)
        veroTrue = diccionario[-1].get(dato[-1])
        countTrue = veroTrue + countTrue
        total = 1 + total
        for col,i in enumerate(dato):
          #incrementamos en uno el valor correspondiente. OJO indice i !!!! controlar numero diccionarios 
          verosimilitudes[veroTrue][i][col] = 1 + verosimilitudes[veroTrue][i][col]

      # calculo de verosimilitudes de clase falso
      for dic in verosimilitudes[0]:
        for key, value in dic.items():
          #dividimos numero de ocurrencias (Atributo y Falso) por el total de (Falso)
          dic[value]= value/(total - countTrue)
       
      # calculo de verosimilitudes de clase true
      for dic in verosimilitudes[0]:
        for key, value in dic.items():
          #dividimos numero de ocurrencias (Atributo y Falso) por el total de (Falso)
          dic[value]= value/(total)
    
  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):
    pass

    
    





  