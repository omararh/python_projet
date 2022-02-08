import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

##########
### Etape 1
### Chargement des données et premières visualisations
############

# Chargement des données d'entraînement dans un objet DataFrame pandas
train_data = pd.read_csv('diabetes_train.csv')
print(train_data) #contenu du fichier sera afficher 

# Affichage du descriptif des données (moyenne , max ..)
print(train_data.describe())

# Visualisation des données - projection selon les axes "Glucose" et "BMI"
## Dans ce dataset, il y a deux classes 1 ou 0, pour la présence ou l'absence de diabète.

list_color= ["green","red"] #les couleurs qu'on va utiliser pour notre graphe 

for i in range(2):
    #loc : Access a group of rows and columns , donc on va prendre tous les colognes avec outcome == i qui seront stocker dans df_subclass
    df_subclass=train_data.loc[train_data["Outcome"] ==i] #1 torunur de boucle : extraire un sous data frame avec outcome = 0 , 2eme..
    plt.scatter(df_subclass["Glucose"],df_subclass["BMI"],color=list_color[i], label=str(i)) 

plt.xlabel("Glucose")
plt.ylabel("BMI")

plt.legend()
plt.show()


##########
### Etape 2
### Prétraitement des données : enlever Nan.. et tous les traitements qu'on a fais dans cette partie 
############

# Remplacement des données manquantes par la moyenne par colonne
#fonction pandas.DataFrame.fillna() permet de remplacer les valeurs de NaN dans DataFrame avec une certaine valeur ! 
train_data = train_data.fillna (value = train_data.mean()) #j'ai des colognes vides(NaN) le but c'est de les remplir par la moyenne de la cologne

# On veut s'assurer que le modèle fonctionnera aussi pour de nouveaux patients. On sépare donc les données en un jeu d'entraînement et un jeu de
# validation.
# On sépare les données de manière à avoir 80 % des données dans l'ensemble d'entraînement et 20 % pour l'ensemble de validation.
# On appelera respectivement ces ensembles diabetes_train et diabetes_valid.

diabetes_train=train_data.sample(frac=0.8,random_state=90)  #on prend 80% des ensembles de donnée
diabetes_valid=train_data.drop(diabetes_train.index) #on prend 20%

# On sépare pour chaque ensemble les features (caractéristiques des personnes) et la cible à prédire ("outcome").
# On notera X_train, Y_train et X_valid, Y_valid respectivement les features et les cibles pour l'ensemble d'entraînement et de validation.
X_train = diabetes_train[diabetes_train.columns[:-1]]
Y_train = diabetes_train[diabetes_train.columns[-1]] #outcome

#Valide represente 20% ..
X_valid = diabetes_valid[diabetes_train.columns[:-1]]
Y_valid = diabetes_valid[diabetes_train.columns[-1]]#outcome

## Affichage des tailles de ces datasets
print(X_train.shape, X_valid.shape)
print(X_valid.shape, Y_valid.shape)

# On renormalise les données de façon à ce que chaque colonne des datasets d'entraînement et de validation soit de moyenne nulle
# et d'écart type 1

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean)/std

print(X_train.mean(axis=0))
print(X_train.std(axis=0))

X_valid = (X_valid - mean)/std

print(X_valid.mean(axis=0))
print(X_valid.std(axis=0))


############
### Etape 3
####Algorithme KPPV: K plus proches voisins
############

# La méthode des K plus proches voisins est un algorithme d'apprentissage supervisé que l'on va utiliser pour classifier les points du jeu de données diabètes.
# À partir d'un point x dont on ne connait pas la classe (Outcome), on cherche les k plus proches voisins de x dans le jeu d'entraînement selon la distance euclidienne.
# On calcule la moyenne de la valeur de Outcome parmi les k plus proches voisins.
# Si elle est inférieure à 0.5, on considère que x appartient à la classe 0 (car le point x a plus de voisins avec la classe 0).
# Sinon, on considère que x appartient à la classe 1.

####  cf. https://fr.wikipedia.org/wiki/Méthode_des_k_plus_proches_voisins
#### https://scikit-learn.org/stable/modules/neighbors.html#classification
##############


# Définition d'une fonction euclidean_distance(v1, v2), calculant la distance euclidienne entre les vecteurs v1 et v2.

def euclidian_distance(v1,v2):

    distance = 0
    for i in range(v1.shape[0]):  #X_train.shape[0] : le nombre de lignes dans X_train
        distance += (v1[i] - v2[i])**2 

    return np.sqrt(distance) 


# Test de la fonction euclidean_distance avec deux points du dataset d'entraînement de façon à vérifier si calcul de distance fonctionne bien.
print(euclidian_distance(X_train.iloc[0],X_train.iloc[1]))

# Fonction qui sélectionne les k plus proches voisins du point x_test dans l'ensemble des points contenus dans X_train.
def neighbors(X_train, y_label, x_test, k):

    list_distances =  []

    ## On calcule toute les distances entre x_test et chaque point de X_train.
    for i in range(X_train.shape[0]):
        distance = euclidian_distance(X_train.iloc[i], x_test)
        list_distances.append(distance)

    ## On trie les poinds de X_train par ordre croissant de distance et on renvoie le dataframe qui contient les k plus proches voisins.
    df = pd.DataFrame()
    df["label"] = y_label
    df["distance"] = list_distances
    df = df.sort_values(by="distance")

    return df.iloc[:k,:]


# Pour le premier point de l'ensemble de validation, affichage de ses 5 plus proches voisins de l'ensemble d'entraînement.
nearest_neighbors = neighbors(X_train, Y_train, X_valid.iloc[0], 5)
print(nearest_neighbors)

## Fonction "prediction" qui prend en entrée un dataframe contenant les plus proches voisins d'un point avec leur classe
# (obtenu par la méthode neighbors) et qui renvoie la classe prédite pour ce point.

def prediction(neighbors): #vu qu'on a que deux vals possible on fais la moyenne , si on avais plus on aurai du compte le nombre de 0 de 1 et de 2 .. et return le plus nombreux 

    ### Pour ce problème de classification à deux classes, au lieu de compter les voisins qui ont la classe 0 ou 1, le plus rapide est de simplement faire la moyenne des labels des voisins.
    ### On sait que cette moyenne sera inférieure à 0.5 s'il y a plus de "0" et supérieure à 0.5 s'il y a plus de "1" parmi les voisins.
    ## Remarque : Pour un problème à plus que deux classes, il faudrait compter les individus dans chaque classe et renvoyer la classe majoritaire.

    mean = neighbors["label"].mean()

    if (mean < 0.5):
        return 0
    else:
        return 1


# Affichage de la prédiction du modèle pour le premier point de l'ensemble de validation
print(prediction(nearest_neighbors))

############
### Etape 4
### Evaluation du modèle
############


# Fonction d'evaluation qui permet de calculer et d'afficher le nombre de vrais positifs, de faux positifs, de vrais négatifs et de faux négatifs
# pour un ensemble de validation donné en entrée. La fonction evaluation calcule également la métrique "accuracy" (ou précision), qui correspond
# à la proportion de prédictions correctes.

#l'idée de voir si on se trompe ou pas 
def evaluation(X_train, Y_train, X_valid, Y_valid, k, verbose=True):

    TP = 0  # prediction égale à 1 et cible égale à 1 (vrai positif, true positive TP)
    FP = 0  # prediction égale à 1 et cible égale à 0 (faux positif, false positive FP)

    TN = 0  # prediction égale à 0 et cible égale à 0 (vrai négatif, true negative TN)
    FN = 0  # prediction égale à 0 et cible égale à 1 (faux négatif, false negative TN)

    total = 0

    for i in range(X_valid.shape[0]):

        nearest_neighbors = neighbors(X_train, Y_train, X_valid.iloc[i], k) 

        if ((prediction(nearest_neighbors) == 1) and (Y_valid.iloc[i] == 1)):
            TP += 1
        elif ((prediction(nearest_neighbors) == 1) and (Y_valid.iloc[i] == 0)):
            FP += 1
        elif ((prediction(nearest_neighbors) == 0) and (Y_valid.iloc[i] == 0)):
            TN += 1
        elif ((prediction(nearest_neighbors) == 0) and (Y_valid.iloc[i] == 1)):
            FN += 1

        total += 1

    accuracy = (TP + TN) / total #LE taux de réussite 

    if verbose:
        print("Vrais positifs : " + str(TP)) # pour concaténer une chaine et un nombre on fais str 
        print("Faux positifs : " + str(FP))
        print("Vrais négatifs : " + str(TN))
        print("Faux négatifs : " + str(FN))

        print("Accuracy:" + str(accuracy))

    return accuracy

### Test des scores obtenus avec l'algorithme des k plus proche voisins sur l'ensemble de validation pour k=5.
evaluation(X_train, Y_train, X_valid, Y_valid, 5)

############
### Etape 5
### Réglage du paramètre "k" du modèle (nombre de voisins). comment on va choisir ce paramétre ? 
############

# Lancement du bloc d'évaluation pour différentes valeurs de k entre 1 et 19 de 2 en 2 et stockage des valeurs d'accuracy obtenues sur l'ensemble de validation
# dans une liste


list_accuracy = []

for k in tqdm(range(1,19,2)):
    list_accuracy.append(evaluation(X_train, Y_train, X_valid, Y_valid, k, verbose=False))

print(list_accuracy)

# # Traçage de la courbe qui montre l'évolution de l'accuracy en fonction de k
# # Quelle valeur de k vous semble la plus pertinente ?
#

x=range(1,19,2)
y=list_accuracy

plt.plot(x,y)
plt.xlabel("k")
plt.ylabel("accuracy")

plt.legend()
plt.show()


############
### Etape 6
###  Application du modèle sur des données de test
############

# Chargement des données de test

X_test = pd.read_csv('diabetes_test_data.csv')
Y_test = pd.read_csv('diabetes_test_target.csv')["Outcome"]

print("X_test")
print(X_test)

print("Y_test")
print(Y_test)

# Remplacement des données manquantes dans les données de test par la moyenne par colonne
X_test = X_test.fillna (value = X_test.mean())

# Normalisation des données de test.
X_test = (X_test - mean)/std

# Calcul de la précision en test du modèle avec le paramètre k=3 qui a été choisi lors de la phase de validation du modèle
evaluation(X_train, Y_train, X_test, Y_test, 3)




############
### Etape 7
### Visualisation du modèle dans le cas où il n'y a que les variables BMI et Glucose (ce qui permet de faire une visualisation en 2D).
############

### Fonction qui permet l'affichage de la frontière de décision du modèle pour une valeur de k donné.

def plot_decision(X, y, k):
    h = 0.2

    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))

    xx0_flat = xx0.ravel()
    xx1_flat = xx1.ravel()

    X_entry = np.stack((xx0_flat, xx1_flat), axis=1)

    y_pred = np.zeros((xx0_flat.shape[0]))

    for i in tqdm(range(X_entry.shape[0])):
        nearest_neighbors = neighbors(pd.DataFrame(X), y, pd.DataFrame(X_entry).iloc[i], k)
        y_pred[i] = prediction(nearest_neighbors)

    preds = y_pred.reshape(xx0.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    plt.figure()
    plt.pcolormesh(xx0, xx1, preds, cmap=cmap_light)
    plt.xlim(xx0.min(), xx0.max())
    plt.ylim(xx1.min(), xx1.max())

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.show()


### Affichage de la frontière de décision du modèle pour k=50.
# On garde seulement deux variables Glucose et BMI pour faire l'affichage et les prédictions.

plot_decision(X_train[["Glucose","BMI"]].values, Y_train, 50)

#### Affichage de la frontière de décision du modèle pour k=1

plot_decision(X_train[["Glucose","BMI"]].values, Y_train, 1)






