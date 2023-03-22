#!/usr/bin/env python
# coding: utf-8

# 
# <pre><h1><center><bold style=" color:'red'">
# ImageMining:
# Atelier N°3: Image Clustering</bold> 
# 
# 
# Hicham Bayad  <br/>
# S148011841  <br/>
#  hicham98bayad@gamil.com</center></h1>
# 
# 
# 
# 
# 
# </pre>

# #<h1>1- Introduction </h1>
# Le clustering consiste en un regroupement (partitionnement) des objets en des groupes (paquets) homogènes (clusters). 
# Un cluster est une collection d’objets (éléments) qui partagent des caractéristiques communes ou trop proches et qui sont dissimilaires aux objets des autres clusters. 
# Le clustering consiste en un regroupement (partitionnement) des objets en des groupes (paquets) homogènes (clusters).<br>
# Un cluster est une collection d’objets (éléments) qui partagent des caractéristiques communes ou trop proches et qui sont dissimilaires aux objets des autres clusters. 

# Les applications du clustering peuvent être répertoriées en 3 catégories :
# 
# 1. Segmentation des bases de données (d’images) en des catégories d’images ; images représentants les forêts, champs, voitures, bâtiments, lacs, … 
# 2. Classification non supervisée
# 3. Extraction de connaissances afin de faire apparaitre des sous-ensembles et sous-concepts éventuellement impossibles à distinguer naturellement.

# # <h1>2- Algorithmes de clustering</h1>
# Il existe une multitude d’algorithmes de clustering qui peuvent être utilisés dans le cas de l’Image mining :
# 
# 1- Les méthodes basées centroïdes telles que les algorithmes des k-moyennes (K-means) ou k-médoïdes <br>
# 2- Les méthodes de regroupement hiérarchique<br>
# 3- Algorithme de maximisation de l'espérance (Expectation-Maximization; EM)<br>
# 5- Algorithme de C-moyennes floues (Fuzzy C-means clustering; FCM) <br>
#    …<br>

# # <h2> 2.1- K-means</h2>
# K-means est l'un des algorithmes les plus connus en classification non supervisée (clustering), simple et largement utilisé.
# 
# Principe
# 
# •	C’est algorithme itératif qui minimise la somme des distances entre chaque pixel et le centroïde de son cluster<br>
# •	Consiste à regrouper les pixels en un certain nombre de classes représentant les régions<br>
# •	Le nombre de classes (K) est choisi selon une connaissance préalable et suivant combien de régions souhaitées<br>
# •	Le critère d’arrêt de l’algorithme est basé sur la stabilité des moyennes <br>

# ## <h3> Avantages </h3>
# 1- L’algorithme de k-Means est très populaire du fait qu’il est très facile à comprendre et à mettre en œuvre, <br>
# 2- La méthode résous une tâche non supervisée, donc elle ne nécessite aucune information sur les données, <<br>
# 3- Rapidité et faibles exigences en taille mémoire, <br>
# 4- La méthode est applicable à tout type de données en choisissant une bonne notion de distance.<br>

# ## <h3> Inconvénients </h3>
# 1- Le nombre de classes est un paramètre de l’algorithme, <br> 
# 2- Un bon choix du nombre k est nécessaire, car un mauvais choix de k produit de mauvais résultats, <br>
# 3- Les points isolés sont mal gérés (Ils doivent appartenir obligatoirement à un cluster ?)  très sensible au bruit,<br>
# 4- L'algorithme du K-Means ne trouve pas nécessairement la configuration la plus optimale correspondant à la fonction objective minimale, <br>
# 5- Les résultats de l'algorithme du K-Means sont sensibles à l'initialisation aléatoires des centres.<br>

# ##<h3> Algorihtme</h3>
# 1- Choix aléatoire de la position initiale des K centroïdes de clusters, <br>
# 2- Affecter les pixels à un cluster suivant un critère de minimisation des distances (généralement selon une mesure de distance euclidienne), <br>
# 3- Une fois tous les pixels placés, recalculer les centroïdes de chaque cluster, <br>
# 4- Réitérer les étapes 2 et 3 jusqu’à ce que plus aucune réaffectation ne soit possible. <br>

# # <h1> 2.2- FCM (Fuzzy C-Means; C-moyennes floues)</h1>
# FCM est un algorithme de classification non-supervisée floue. Son principe est le suivant: <br>
# - Un pixel peut appartenir à plusieurs classes mais selon le degré d’appartenance,<br>
# - Chaque degré exprime l’appartenance incertaine d’un pixel à  une région donnée, <br>
# - Le degré d’appartenance se situe dans l’intervalle [0, 1] et les classes obtenues ne sont pas forcément disjointes.<br>

# ## <h3> Avantages</h3>
# 1- Non-supervisé <br>
# 2- Converge toujours

# ## <h3> Inconvénients </h3>
# 1- Le principal inconvénient de ces méthodes réside dans le fait que la classification ne repose que sur l’intensité des éléments de volume sans prendre en compte l’information spatiale, ce qui la rend sensible au bruit et a l’hétérogénéité d’intensité, <br>
# 2- Longue durée de calcule, <br>
# 4- Sensibilité a la conjecture initiale (vitesse, minimums locaux,bruit). <br>

# ## <h3> Algorithme </h3>
# 1- Fixer les paramètres : C : le nombre de classes, m: degré de flou, ε : critère d’arrêt, <br>
# 2- Initialiser la matrice de degrés d’appartenance U par des valeurs aléatoires dans l’intervalle [0,1], <br>
# 3- Calculer les centroïdes des classes, <br>
# 4- Mettre à jour la matrice degrés d’appartenance suivant la position des centroïdes, <br>
# 5- Répéter les étapes 3 à 4 jusqu’à satisfaction du critère d’arrêt.<br>

# # <h2> 2.3- Expectation Maximisation (EM)</h2>
# L'Expectation Maximisation (EM) est utilisé pour trouver l’estimation du maximum de vraisemblance à partir d’un ensemble de données. Le choix initial des paramètres affecte le déroulement de la classification jusqu’à générer des clusters incohérents.<br>
# 

# ## <h3>Algorithme</h3>
# 1- Initialisation : Choisir des paramètres initiaux : <br>
# 
# > μj : la moyenne de chaque cluster,<br>
# > σ : la variance , <br>
# > P(Cj) : la proportion de chaque classe. <br>
# 
# 2- Étape E : Estimer la probabilité que le vecteur x des pixels appartient à un cluster,<br> 
# 3- Étape M: Mise à jour les paramètres, <br>
# 4- Répéter l'itération de base, jusqu'à stabilisation des paramètres. <br>

# # <h2>2.4- Mean-Shift</h2>
# Le Mean-Shift est une méthode de clustering basée sur l’estimation de densité. <h3>Principe</h3>
# Chaque pixel est représenté par un point dans l’espace à N dimensions pour ce faire, on calcule la densité locale en un point et on déplace itérativement la fenêtre en direction du gradient de densité maximum.

# ## <h3>Algorithme</h3>
# 1- Choisir une répartition uniforme des fenêtres de recherche initiales,<br>
# 2- Calculer le centroide des données pour chaque fenêtre, <br>
# 3- Centrer la fenêtre de recherche sur le centroide de l'étape 2,<br> 
# 4- Répéter les étapes 2 et 3 jusqu'un à convergence, <br>
# 5- Fusionner les fenêtres se trouvant au même point final,<br> 
# 6- Grouper les données traversées par les fenêtres fusionnées.<br>

# # <h1> 3- Implémentation </h1>
# En ImageMining, nous avons deux façons pour appliquer (utiliser) le Clustering.
#   
# 
# 1.   Regrouper une base d’images en des sous-ensembles. Par exemple, nous avons une base d'image contenant une multitude de types d'images; images aériennes, images des animaux, images des vehicules, ... et on veut regrouper chacune des images de la base dans sa catégoorie.
# 2.   Segmenter des images. 
# 
# 
# Dans cet atelier, nous allons tout d'abord faire le clustering de deux bases d'images<br>
# 
# 1. Images à regrouper en 2 classes<br>
# 2. Image à regrouper en 4 classes<br>
# 
# Le resultats du clustering sera enregistré dans un fichier *.csv en respectant la forme suivante:
# 
# image_name,class
# image1.jpg,1
# image2.jpg,2
# image3.jpg,1
# 
# Par la suite nous allons segmenter des images utilisant un algorithme de clustering et aussi nous allons appliquer la classification supervisée pour la segmentation d'images. 

# ## <h2>3.1- Clustering d'une base d'images en 2 classes</h2>
# L'objectif est de regrouper les images de la base d'images "2Classes" en 2 classes. Visuellement, les images peuvent étre regouées en les classes "Cartes" et "Nature". 
# Il faut developper un modéle capable de séparer les images correctement.

# ###<h3>3.1.1- Extraction de caractéristiques (features extraction)</h3> 
# 
# Il est très pratique d'utiliser les caractéristiques d'une image à la place des valeurs d'intensité de chaque pixel dans un processus de classification.  
# 
# Ainsi, la première étape du ce processus et l’extraction de caractéristiques. Dans cet atelier, 81 caractéristiques a utiliser et qui sont : 
# 
# - Les moments statistiques (moyenne et standard déviation) dans l'espace R, G et B ==> 6 caractéristiques 
# 
# - l'histogramme quantifié (8,2,2) cumulatif dans l'espace HSV ==> 32 caractéristiques 
# 
# - Le Contraste, la Corrélation, l'Energie el l'Homogénéité calculées à partir de la matrice GLCM pour la texture ==> 4 caractéristiques 
# 
# - L'histogramme du vecteur LBP sur l'image à niveau de gris pour la caractérisation de la texture ==> 32 caractéristiques 
# 
# - Les moments de Hu pour la caractérisation de la forme ==> 7 caractéristiques 

# In[1]:


# Variables globales 
# color_Moments 6 
# hsvHistogramFeatures 32 
# textureFeatures 4 
# shapeFeatures 7 
# lbp_histogram 32 
# Total 81 
#featuresNumber=82 

# pour les bases d'images en 2 et 4 Classes 
# Télécharger les bases d'images en cliquant sur le lien suivant : 
""" 
https://drive.google.com/drive/folders/1zPDohqJL7vb_toVEWBmfO7UuhZd_HaNN?usp=sharing 

Il faut copier le dossier DB_Clustering sur votre Drive.  
Et ne pas oublier d'autoriser l'accès au Drive 
 
n_clusters=2 
path2="/content/drive/MyDrive/DB_Clustering/2Classes" 

# pour 4 Classes 
#path4="/content/drive/MyDrive/DB_Clustering/4Classes" 

from google.colab import drive 
drive.mount('/content/drive') """


# In[2]:


import numpy as np
import cv2
import math

from skimage.feature import greycomatrix , greycoprops
from skimage import io , color,img_as_ubyte

from os import listdir
from matplotlib import image

import os

#LBP
from skimage import feature

import matplotlib.pyplot as plt


# <h3>On utilisant les fonction qui deja défini dans l'atelier CBIR </h3>
# <h4>- Color_Moments 6 </h4>

# In[3]:


def color_moments(img):
    colorFeatures=[]
    for i in range(3):
        colorFeatures.append(np.mean(img[ :, : , i ]))
        colorFeatures.append(np.std( img[ :, : , i ]))
    return colorFeatures
    


# <h4>- HsvHistogramFeatures 32  </h4>

# In[4]:


def getHsvHistogramFeatures(img):
    
    #featurHsvHistogramFeatures=[]
    imgHsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    # [0,1,2] :  les canaux qui calcule
    # [8,2,2] :   Nombre de bainse de chaque canaux
    # [0,360,0,255,0,255] : L'intensité de chaque canaux 
    hist_hsv=cv2.calcHist(imgHsv ,[0,1,2] ,None, [8,2,2],[0,360,0,255,0,255]) #Histogramme Normaliser
    
    return hist_hsv.flatten()
    


# <h4>- ShapeFeatures 7 </h4>

# In[5]:


def increaseValueMoments(moments):
    for i in range(0,7):
        moments[i]=-1*math.copysign(1.0 , moments[i])*math.log10(abs(moments[i]))
    
    return moments

def get7Moment(img):
    img_gray=cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    ret,img_binair=cv2.threshold( img_gray, 100, 255, cv2.THRESH_BINARY)
    moments=cv2.HuMoments(cv2.moments(img_binair)).flatten()
    
    return increaseValueMoments(moments)
    


# <h4>- TextureFeatures 4 </h4>

# In[6]:


# Calculate texture properties of a GLCM

def contrast_feature(matrix_coocurrence):
    contrast = greycoprops(matrix_coocurrence, 'contrast')
    return  contrast

def dissimilarity_feature(matrix_coocurrence):
    dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')    
    return  dissimilarity

def homogeneity_feature(matrix_coocurrence):
    homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
    return  homogeneity

def energy_feature(matrix_coocurrence):
    energy = greycoprops(matrix_coocurrence, 'energy')
    return  energy

def correlation_feature(matrix_coocurrence):
    correlation = greycoprops(matrix_coocurrence, 'correlation')
    return  correlation


def entropy_feature(matrix_coocurrence):
    entropy = greycoprops(matrix_coocurrence, 'entropy')
    return "Entropy = ", entropy


def getFeaturesTexturGLCM(img):
    freaturesTexturs=[]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Extraction de matrice co-ocurrence de (distance egale '1' et de angle egale '0')
    matrix_coocurrence = greycomatrix(gray, [1], [0], levels=256, normed=False, symmetric=False) 
    
    freaturesTexturs.append(contrast_feature(matrix_coocurrence)[0][0])
    freaturesTexturs.append(homogeneity_feature(matrix_coocurrence)[0][0])
    freaturesTexturs.append(energy_feature(matrix_coocurrence)[0][0])
    freaturesTexturs.append(correlation_feature(matrix_coocurrence)[0][0])
    
    return freaturesTexturs


# <h4>- LBP_histogram 32 </h4>

# In[7]:


def featuresLBPHist(img):
    numPoint=8
    raduis=3
    
    img_gray=cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    
    lbp=feature.local_binary_pattern(img_gray ,numPoint ,raduis,method="default" ).astype('uint8')
    #print("==> ", np.shape(lbp))
    #print(type(lbp[0][0]))
    #plt.imshow(lbp)
    #plt.title("LBP_image")    
    hist_ref=cv2.calcHist(lbp ,[0],None , [32], [0,255])
    return hist_ref.flatten()


# <h4> Code pour l'extraction des caracéristiques pour une image</h4>

# In[8]:


# color_Moments 6 
# hsvHistogramFeatures 32 
# textureFeatures 4 
# shapeFeatures 7 
# lbp_histogram 32 
# Total 81 

def getFeatures(img):
    
    return np.concatenate((color_moments(img) , getHsvHistogramFeatures(img) , get7Moment(img) , getFeaturesTexturGLCM(img) , featuresLBPHist(img)))
    


# <h4>- Lecture des images</h4>

# <h3>chargées les images et extraire les caractéristique a chaque image , charge tous les caractéristiques des un tableaux </3>

# In[9]:


"""
  Input: dataset_path: nom du dossier contenant les images par dossier (classe)
  Outpurs:
  features: Matrice de caractéristiques de taille (nombre_d_image x nombre_features)
  image_names: Vecteur retournant les noms des images de la base. 
"""
def createFeatures(path_dataSet):
    
    imageNames=[]
    features=[]
    dirs=os.listdir(path_dataSet)
    dirs.sort()
    
    for image_name in dirs:
        image_path=path_dataSet+"/"+image_name
        image=cv2.imread(image_path)
        feature=getFeatures(image)
        imageNames.append(image_path)
        features.append(feature)
    return imageNames,features ,image 


# In[10]:


path2C = "Db_clustring\\2Classes"
imgs_names,imgs_features , img = createFeatures(path2C)


# In[11]:


print("dim featuresLBPHist", np.shape(featuresLBPHist(img)) )
print("dim getFeaturesTexturGLCM",np.shape(getFeaturesTexturGLCM(img)))
print("dim get7Moment",np.shape(get7Moment(img )))
print("dim getHsvHistogramFeatures",np.shape(getHsvHistogramFeatures(img)))
print("dim color_moments",np.shape(color_moments(img)))
print("la dimension de vecteur caractéristique : " , np.shape(imgs_features)) #200 images et 81 caractéristiges a chaque image  


# In[12]:


plt.imshow(img)


# In[13]:


imgs_features


# <h2>3.1.2 Clustering avec le Kmeans</h2>
# Nous allons appliquer l'algorithme Kmeans sur la matrice de caractéristiques

# In[14]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
n_clusters=2
kmeans = KMeans(n_clusters=n_clusters, algorithm="elkan", n_init=100, init='random')
"""""
le code transforme les fonctionnalités d’entrée à l’aide de la et retourne un nouveau tableau d’entités transformées. 
Les entités transformées auront la même forme que les entités d’entrée, mais seront normalisées pour suivre une
distribution spécifiée avec un nombre fixe de quantiles.QuantileTransformer
"""""
img_features = QuantileTransformer(n_quantiles=10).fit_transform(imgs_features)
kmeans.fit(img_features)
print(kmeans.labels_)


# importation des résultats dans un fichier *.csv en respectant la forme :
# Image_name,Class
# nom_de_l_image1,classe_prédite1
# nom_de_l_image2,classe_prédite2
# ....

# In[15]:


import pandas as pd

def create_csv( nameImage ,predLbl  ,nameFile  ):
    data ={
        'imageName':nameImage,
        'Class' :predLbl
    }
    df = pd.DataFrame(data , columns=[ 'imageName','Class' ] )
    df.to_csv(nameFile , header=True , index=False)


# In[16]:


create_csv(imgs_names , kmeans.labels_ , 'cluster_Kmeans.csv')


# En vérifiant le résultat de clustering, nous avons obtenu un taux égal à 91.5% avec l’algorithme Kmeans

# ### <h3>-3.1.2 KMedoids</h3>
# Dans cette partie, nous allons appliquer l'algorithme de classification non-supervisée KMedoids sur la mème matrice de caractéristiques

# In[17]:


# Il faut installer scikit-learn-extra pour pouvoir utiliser KMedoids
#!pip install scikit-learn-extra


# In[18]:


from sklearn_extra.cluster import KMedoids
kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
kmedoids.fit(imgs_features)


# importation des résultats:

# In[19]:


create_csv(imgs_names, kmedoids.labels_,'cluster_KMedoids.csv')


# Après vérification des résultats de clustering, le KMedoids nous a permis d'avoir un taux égal à 86.5%

# <h3>3.1.3- Clustering Hierarchique</h3>
# Nous allons utiliser un clustering hiérarchique en se basant sur l'algorithme AgglomerativeClustering utilisant la mème matrice de caractéristiques

# In[20]:


from sklearn.cluster import AgglomerativeClustering

hc_agg = AgglomerativeClustering(n_clusters=n_clusters)
hc_agg.fit(imgs_features)


# In[29]:


hc_agg.labels_


# Importation des résultats

# In[21]:


create_csv(imgs_names, hc_agg.labels_,'cluster_Hierarchical.csv')
print(np.array(hc_agg.labels_).shape)


# Aprés vérification des résultats de clustering, le Clustering Hierarchique nous a permis d'avoir un taux égal à 91.5%

# <h3>3.1.4 FCM <h3>
# Pour une 4éme implémentation du clustering, nous allons utiliser l'algorithme FCM

# In[22]:


#!pip install -U scikit-fuzzy
#import skfuzzy as fuzz
#!pip install fuzzycmeans


# In[30]:


get_ipython().system('pip install scikit-fuzzy')


# In[74]:


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# convertir les données a numpy.array
data = np.array(imgs_features)

# Définir les paramètres FCM
# Définir le nombre de clusters
n_clusters = 2

# Définir la marge d'erreur maximale
epsilon = 0.001

# Définir le nombre maximal d'itérations
max_iter = 1000


# In[75]:


# Initialiser les centres des clusters aléatoirement
centers = np.random.uniform(-1, 1, (n_clusters, 200))


# In[76]:


# Appliquer l'algorithme FCM
cntr, u, a, b, c, d, e = fuzz.cluster.cmeans(
    data.T, n_clusters, m=2, error=epsilon, maxiter=max_iter, init=centers)


# Afficher les centres des clusters
#print(cntr)

# Obtenir les labels
fcm_labels = np.argmax(u, axis=0)

# Afficher l'appartenance de chaque point à chaque cluster
#print(u)

# Afficher les labels
print(fcm_labels)


# In[77]:


np.shape(data)


# Importation des résultats

# In[78]:


create_csv(imgs_names, fcm_labels,'cluster_FCM.csv')


# Aprés vérification des résultats de clustering, le FCM nous a permis d'avoir un taux égal à 85.5%

#  <h1>3.2- Clustering en 4 classes </h1> <br>
# Dans cette partie nous allons faire un clustering de la base d'image "4Classes" en 4 classes. Nous avons opter pour l'algorithme de clustering K-Means et nous allons sauvegarder le résultat dans un fichier *.csv. 

# In[82]:


# Extraction des caractéristiques
path4C="Db_clustring\\4Classes"
imgs_names,imgs_features , img=createFeatures(path4C)

# Clusetring
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
n_clusters=4
kmeans = KMeans(n_clusters=n_clusters, algorithm="elkan", n_init=100, init='random')
features = QuantileTransformer(n_quantiles=10).fit_transform(imgs_features)
kmeans.fit(features)
print(kmeans.labels_)


# In[85]:


# Importation du résultat
create_csv(imgs_names, kmeans.labels_,'cluster_Kmeans4C.csv')


# In[ ]:





# In[ ]:




