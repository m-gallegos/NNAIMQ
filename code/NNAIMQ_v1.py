############################################################################
                    #    #  #    #    ##     #####   ####
                    # #  #  # #  #   #  #      #    #    #
                    #  # #  #  # #  ######     #    #    #
                    #    #  #    # #      #  #####   #### #
#---------------------------------------------------------------------------
# Original NNAIMQ code (V 1.0), 2021, University of Oviedo
# Author (s): M. Gallegos in collaboration with J.M. Guevara-Vela and
# A. M. Pendas.
############################################################################
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from random import randint

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

############################################################################
### Get where the executables are located, ROOT_DIR
### Get where the input and xyz files are located, CURR_DIR

absolutepath = os.path.abspath(__file__)
dividido = absolutepath.split("/")
ROOT_DIR = "/".join(dividido[:-1])

CURR_DIR = os.getcwd()

############################################################################

def norm(x,mean,std):
  y=np.empty_like(x)
  y[:]=x
  y=np.transpose(y)
  counter=0
  for i in y:
    y[counter,:]=(y[counter,:]-mean[counter])/(std[counter])
    counter=counter+1
  y=np.transpose(y)
  return y

############################################################################

input_largo=sys.argv[1]
dividido = input_largo.split("/")
list_geom=CURR_DIR+"/"+dividido[-1]
print("Reading the geometry files from",list_geom)

############################################################################

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)

############################################################################

os.chdir(ROOT_DIR)

mean_C= np.loadtxt("nnqC.mean",dtype='f')
std_C = np.loadtxt("nnqC.std",dtype='f')
mean_H= np.loadtxt("nnqH.mean",dtype='f')
std_H = np.loadtxt("nnqH.std",dtype='f')
mean_O= np.loadtxt("nnqO.mean",dtype='f')
std_O = np.loadtxt("nnqO.std",dtype='f')
mean_N= np.loadtxt("nnqN.mean",dtype='f')
std_N = np.loadtxt("nnqN.std",dtype='f')
print("Loading data statistics............DONE!")

############################################################################

model_C = tf.keras.models.load_model('nnqC.h5')
model_H = tf.keras.models.load_model('nnqH.h5')
model_O = tf.keras.models.load_model('nnqO.h5')
model_N = tf.keras.models.load_model('nnqN.h5')
print("Loading Neural Networks............DONE!")

############################################################################

col_c = 129
col_h = 132
col_o = 111
col_n = 112
colname_c=['AtomNum']
colname_h=['AtomNum']
colname_o=['AtomNum']
colname_n=['AtomNum']
for i in range(1,col_c+1):
    colname_c.append('g' + str(i))
for i in range(1,col_h+1):
    colname_h.append('g' + str(i))
for i in range(1,col_o+1):
    colname_o.append('g' + str(i))
for i in range(1,col_n+1):
    colname_n.append('g' + str(i))
column_names_c=colname_c
column_names_h=colname_h
column_names_o=colname_o
column_names_n=colname_n

############################################################################

with open(list_geom) as f34:
    for geomline in f34:
        geom=geomline.rstrip('\n') 
        geom=geom.replace(' ', '') 
        print("************************************************************")
        print("Computing the ACSF descriptor for file", geom)
        print("************************************************************")
        geom=CURR_DIR+"/"+geom
        f=open(geom,"r")
        contents=f.read()
        f.close()
        
        size=len(geom)
        nombre=geom[:size-4]
        subprocess.check_call([r"./SSFC.exe", geom, nombre])
        
        acsf_list=[]
        acsf_list.append(nombre + ".acsf")
        acsf_list.append(nombre + "_C.acsf")
        acsf_list.append(nombre + "_H.acsf")
        acsf_list.append(nombre + "_O.acsf")
        acsf_list.append(nombre + "_N.acsf")
        for i in acsf_list:
            #print(i)
            with open(i,'r+') as fopen:
                string = ""
                for line in fopen.readlines():
                    string = string + line[:-2] + "\n"
            
            with open(i,'w') as fopen:
                fopen.write(string)
        ######################################################
        # C ATOMS
        ######################################################
        print("Starting prediction of C atoms")
        if (os.stat(acsf_list[1]).st_size != 0) :
           raw_C = pd.read_csv(acsf_list[1],names=column_names_c,na_values="?",
                                     comment='\t',sep=",",skipinitialspace=True)
           dataset_C = raw_C.copy()
           dataset_C = dataset_C.dropna()
           data_C=dataset_C
           data_C_stats=data_C.describe()
           data_C_stats.pop("AtomNum")
           data_C_stats=data_C_stats.transpose()
           data_C_labels=data_C.pop('AtomNum')
           normed_C=norm(data_C,mean_C,std_C)
           C_predictions=model_C.predict(normed_C).flatten()
        ######################################################
        # H ATOMS
        ######################################################
        print("Starting prediction of H atoms")
        if (os.stat(acsf_list[2]).st_size != 0) :
           raw_H = pd.read_csv(acsf_list[2],names=column_names_h,na_values="?",
                                     comment='\t',sep=",",skipinitialspace=True)
           dataset_H = raw_H.copy()
           dataset_H = dataset_H.dropna()
           data_H=dataset_H
           data_H_stats=data_H.describe()
           data_H_stats.pop("AtomNum")
           data_H_stats=data_H_stats.transpose()
           data_H_labels=data_H.pop('AtomNum')
           normed_H=norm(data_H,mean_H,std_H)
           H_predictions=model_H.predict(normed_H).flatten()
        ######################################################
        # O ATOMS
        ######################################################
        print("Starting prediction of O atoms")
        if (os.stat(acsf_list[3]).st_size != 0) :
           raw_O = pd.read_csv(acsf_list[3],names=column_names_o,na_values="?",
                                     comment='\t',sep=",",skipinitialspace=True)
           dataset_O = raw_O.copy()
           dataset_O = dataset_O.dropna()
           data_O=dataset_O
           data_O_stats=data_O.describe()
           data_O_stats.pop("AtomNum")
           data_O_stats=data_O_stats.transpose()
           data_O_labels=data_O.pop('AtomNum')
           normed_O=norm(data_O,mean_O,std_O)
           O_predictions=model_O.predict(normed_O).flatten()
        ######################################################
        # N ATOMS
        ######################################################
        print("Starting prediction of N atoms")
        if (os.stat(acsf_list[4]).st_size != 0) :
           raw_N = pd.read_csv(acsf_list[4],names=column_names_n,na_values="?",
                                     comment='\t',sep=",",skipinitialspace=True)
           dataset_N = raw_N.copy()
           dataset_N = dataset_N.dropna()
           data_N=dataset_N
           data_N_stats=data_N.describe()
           data_N_stats.pop("AtomNum")
           data_N_stats=data_N_stats.transpose()
           data_N_labels=data_N.pop('AtomNum')
           normed_N=norm(data_N,mean_N,std_N)
           N_predictions=model_N.predict(normed_N).flatten()
        ########################################################
        # MOLECULAR CHARGE
        ########################################################
        molec_charge=0.0
        if (os.stat(acsf_list[1]).st_size != 0) :
         for i in C_predictions:
            molec_charge=molec_charge+i
        if (os.stat(acsf_list[2]).st_size != 0) :
         for i in H_predictions:
            molec_charge=molec_charge+i
        if (os.stat(acsf_list[3]).st_size != 0) :
         for i in O_predictions:
            molec_charge=molec_charge+i
        if (os.stat(acsf_list[4]).st_size != 0) :
         for i in N_predictions:
            molec_charge=molec_charge+i
        print("Total Molecular Charge", molec_charge)
        #########################################################
        # OUTPUT FILES
        #########################################################
        cargas_output=nombre+".charge"
        with open(geom) as f:
             lineas = f.readlines()[2:]
        etiqueta=[]
        for line in lineas:
            valores=line.split()
            etiqueta.append(valores[0])
        vector=[]
        contador=0
        contador_c=0
        contador_h=0
        contador_o=0
        contador_n=0
        for i in  etiqueta:
            if (i == "C"):
              vector.append(C_predictions[contador_c])
              contador_c+=1
            elif (i=="H"):
              vector.append(H_predictions[contador_h])
              contador_h+=1
            elif (i=="O"):
              vector.append(O_predictions[contador_o])
              contador_o+=1
            elif (i=="N"):
              vector.append(N_predictions[contador_n])
              contador_n+=1
            contador+=1
        contador=0
        vector=np.array(vector)
        contador=0
        num=1
        with open(cargas_output, 'w') as fout:
             fout.write(" Atom Number " + " Atom Label " + " Charge " + "\n")
             for i in etiqueta:
                fout.write(str(num) + "  " + i + "  " +  str(vector[contador]) + "\n")
                contador+=1
                num+=1
