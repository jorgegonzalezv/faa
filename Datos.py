#!/usr/bin/python
# -*- coding: utf-8 -*-

#Autores: Javier Galan Sanchez y Jorge Gonzalez Villacañas

import numpy as np

TiposDeAtributos=('Continuo','Nominal')

class Datos(object):
 
	def __init__(self, nombreFichero):

		#La declaracion de las variables (listas) hemos decidido que sea local ya que cada objeto de la clase Datos, dependiendo
		#del fichero de entrada, tendra unos valores u otros
		self.nombreFichero=nombreFichero
		self.tipoAtributos=[]
		self.nombreAtributos=[]
		self.nominalAtributos=[]
		self.datos=[]
		self.diccionarios=[]
		self.numInstancias=0

		#llamada a funcion para no meter todo el codigo en el constructor
		self.leerFichero()

	def leerFichero(self):

		#Abrimos el fichero en modo lectura
		f=open(self.nombreFichero, 'r')

		#Obtenemos el numero de instancias y convertimos a numero entero
		self.numInstancias=int(f.readline())

		#Obtenemos los nombres de los atributos
		self.nombreAtributos=f.readline().rstrip('\n')
		self.nombreAtributos=self.nombreAtributos.split(',')

		#Inicializamos tipoAtributos con el tipo de atributo de cada variable (Continuo o Nominal)
		self.tipoAtributos=f.readline().rstrip('\n')
		self.tipoAtributos=self.tipoAtributos.split(',')
		#Generamos la lista booleana nominalAtributos a partir de tipoAtributos
		for tipo in self.tipoAtributos:
			if tipo==TiposDeAtributos[1]:
				self.nominalAtributos.append(True)
			elif tipo==TiposDeAtributos[0]:
				self.nominalAtributos.append(False)

		for i in range(self.numInstancias):
			fila_i = f.readline().rstrip('\n')
			fila_i = fila_i.split(',')

			#Controlamos que los valores sean del tipo Continuo o Nominal segun lo que corresponda
			#Vamos insertando cada fila en la matriz datos tras cada iteracions	
			for j in range(len(self.tipoAtributos)):
				if (self.tipoAtributos[j] == 'Continuo'):
					try:
						#Casteamos los atributos continuos a float pues sera util para el futuro
						#print(fila_i)
						fila_i[j] = float(fila_i[j])
					except ValueError:
						print("El tipo del atributo debe ser CONTINUO y no lo es.")
						f.close()
						raise
				elif (self.tipoAtributos[j] == 'Nominal'):
					try:
						fila_i[j] = str(fila_i[j])
					except ValueError:
						print("El tipo del atributo debe ser NOMINAL y no lo es.")
						f.close()
						raise

			#Anadimos la fila a la matriz de datos
			self.datos = np.append(self.datos, fila_i)

		#Reajustamos el numero de filas y columnas para generar la matriz con toda la informacion
		self.datos = np.reshape(self.datos, (self.numInstancias, len(self.nombreAtributos)))

		
		#print(type(self.datos[0, 1]))

		#Creacion de variables auxiliares para el siguiente bucle
		lista = []
		dic = dict()

		for i in range(len(self.tipoAtributos)):
			if self.nominalAtributos[i] == True:
				for j in range(self.numInstancias):
					lista.append(self.datos[j][i])
				#Ordenamos la lista de claves para que cumpla con el orden lexicografico pedido
				lista.sort()
				#La funcion zip nos permite crear un diccionario a partir de dos iterables de la misma longitud
				dic = dict(zip(lista, range(self.numInstancias)))
				dic = dict(zip(dic.keys(), range(len(dic))))
				#Insertamos el diccionario en la lista
				self.diccionarios = np.append(self.diccionarios, dic)
				#Vaciamos la lista auxiliar para poder usarla en la siguiente iteracion
				del lista[:]
			else:
				#Si el atributo es continuo, insertamos un diccionario vacio
				self.diccionarios = np.append(self.diccionarios, dict())

		f.close()

	def printDatos(self):
		print('Nombre fichero: ' + self.nombreFichero + "\n")
		print("Nombre de los atributos: \n",self.nombreAtributos, "\n")
		print("Tipo de los atributos: \n",self.tipoAtributos, "\n")
		print("Tipo de los atributos (True/False): \n",self.nominalAtributos, "\n")
		print("Matriz de datos: \n",self.datos, "\n")
		print("Diccionarios: \n", self.diccionarios, "\n\n")


	#Funcion que recibirá una lista de índices correspondientes a un subconjunto de
	#patrones a seleccionar sobre el conjunto total de datos. El método devolverá la submatriz de
	#datos que corresponde a los índices pedidos.
	def extraeDatos(self,idx):
		return self.datos[idx, :]




		