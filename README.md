# ImageMinig_Clustering

Le clustering consiste en un regroupement (partitionnement) des objets en des groupes (paquets) homogènes (clusters). 
Un cluster est une collection d’objets (éléments) qui partagent des caractéristiques communes ou trop proches et qui sont dissimilaires aux objets des autres clusters. 
Le clustering consiste en un regroupement (partitionnement) des objets en des groupes (paquets) homogènes (clusters).<br>
Un cluster est une collection d’objets (éléments) qui partagent des caractéristiques communes ou trop proches et qui sont dissimilaires aux objets des autres clusters. 


<h1> - Implémentation </h1>
En ImageMining, nous avons deux façons pour appliquer (utiliser) le Clustering.
  

1.   Regrouper une base d’images en des sous-ensembles. Par exemple, nous avons une base d'image contenant une multitude de types d'images; images aériennes, images des animaux, images des vehicules, ... et on veut regrouper chacune des images de la base dans sa catégoorie.
2.   Segmenter des images. 


Dans cet atelier, nous allons tout d'abord faire le clustering de deux bases d'images<br>

1. Images à regrouper en 2 classes<br>
2. Image à regrouper en 4 classes<br>

Le resultats du clustering sera enregistré dans un fichier *.csv en respectant la forme suivante:

image_name,class
image1.jpg,1
image2.jpg,2
image3.jpg,1

Next we will segment images using a clustering algorithm and also we will apply supervised classification for image segmentation. 


# <h1>The Clustering Algorithms Used</h1>
Il existe une multitude d’algorithmes de clustering qui peuvent être utilisés dans le cas de l’Image mining :

1- Les méthodes basées centroïdes telles que les algorithmes des k-moyennes (K-means) ou k-médoïdes <br>
2- Les méthodes de regroupement hiérarchique<br>
3- Algorithme de maximisation de l'espérance (Expectation-Maximization; EM)<br>
5- Algorithme de C-moyennes floues (Fuzzy C-means clustering; FCM) <br>
   …<br>

Translated with www.DeepL.com/Translator (free version)
