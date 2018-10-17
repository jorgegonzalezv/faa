from abc import ABCMeta,abstractmethod
import math
import numpy as np
import random

class Particion:
  
  #indicesTrain=[]
  #indicesTest=[]
  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]
  def print(self):
    print('Indices Train: '+str(self.indicesTrain))
    print('Indices Test: '+str(self.indicesTest) + "\n")


#####################################################################################################

class EstrategiaParticionado:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta
  nombreEstrategia="null"
  numeroParticiones=0
  particiones=[]
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

  nombreEstrategia = "ValidacionSimple"

  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  def __init__(self,porcentaje,numeroParticiones):
    self.porcentaje = porcentaje
    self.numeroParticiones = numeroParticiones
    self.particiones = []
	
  def creaParticiones(self,datos,seed=None):  

    #inicio semilla random  
    random.seed(seed)

    #calculo del total de elementos en el conjunto de entrenamiento
    instaciasTrain = math.ceil(self.porcentaje * datos.numInstancias);

    #generacion de tantas particiones como especificadas
    for i in range(self.numeroParticiones):
      particion = Particion()

      #generamos una permutacion aleatoria de los indices
      permutacion = np.random.permutation(datos.numInstancias)

      #teniendo el cuenta el porcentaje deseado distribuimos los indices
      particion.indicesTrain = permutacion[:instaciasTrain]
      particion.indicesTest = permutacion[instaciasTrain:]  
        
      #insertamos en la lista de particiones
      self.particiones.append(particion)

    return self.particiones


      
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones
  # y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  nombreEstrategia = "ValidacionCruzada"

  def __init__(self,k):
    self.k = k
    self.particiones = []


  def creaParticiones(self,datos,seed=None):
    random.seed(seed)

    tamano = math.floor(datos.numInstancias/self.k)

    #generamos una permutacion aleatoria de los indices
    permutacion = np.random.permutation(datos.numInstancias)

    resto = datos.numInstancias%self.k
    if resto: restos = permutacion[(tamano-resto):]


    for i in range(self.k):
      particion = Particion()

      if resto > 0: salto = tamano + 1
      else: salto = tamano

      particion.indicesTest = permutacion[i*salto:(i+1)*salto]
      particion.indicesTrain = np.concatenate((permutacion[:i*salto], permutacion[(i+1)*salto:]))

      resto = resto - 1


      #particion.indicesTest = permutacion[i*tamano:(i+1)*tamano]
      #particion.indicesTrain = np.concatenate((permutacion[:i*tamano], permutacion[(i+1)*tamano:]))

      #resto = resto -1


      #if (resto!=0):
      #  if (i<resto):
      #    particion.indicesTest.insert(restos[i])
      #    particion.indicesTrain.remove(restos[i])
      
      #insertamos en la lista de particiones
      self.particiones.append(particion)

    return self.particiones

    
#####################################################################################################

class ValidacionBootstrap(EstrategiaParticionado):

  # Crea particiones segun el metodo de boostrap
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):    
    random.seed(seed)
    pass
    
