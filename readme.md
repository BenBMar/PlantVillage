# Projet Deep Learning – Vision par ordinateur

## 1. Description
Ce projet a été réalisé dans le cadre du module **Intelligence Artificielle**, 

séance 1 : Vision par ordinateur.  

**Objectif :** L'objectif de ce projet est de développer et d'entraîner un modèle de Deep Learning, spécifiquement un Réseau de Neurones Convolutifs (CNN), capable de classifier automatiquement l'état de santé de feuilles de plantes à partir d'images, en utilisant le jeu de données public PlantVillage. La finalité est de créer un outil d'aide à la décision pour la détection précoce des maladies, contribuant ainsi à une agriculture plus durable et productive.

## 2. Contexte
**Secteur :** Agriculture de précision (AgriTech)  
**Problématique :** Les maladies des plantes sont une cause majeure de pertes de rendement mondiales. Le diagnostic manuel par des experts est lent, coûteux et difficilement accessible à grande échelle. Ce projet s'attaque à cette problématique en proposant une solution automatisée et précise basée sur la vision par ordinateur pour identifier les maladies foliaires rapidement.

## 3. Données

-   **Dataset :** [Plant Disease Dataset (Subset)](https://www.kaggle.com/datasets/emmarex/plantdisease)
    -   Les données ont été chargées directement depuis Kaggle en utilisant la fonction `tf.keras.utils.image_dataset_from_directory`, qui interprète la structure des dossiers comme des labels de classe.

-   **Taille et Composition :**
    -   **Nombre total d'images :** Environ 20 640 images couleur au format JPG.
    -   **Nombre de classes :** 15 classes distinctes.
    -   **Espèces de plantes couvertes :** Poivron (Pepper), Pomme de terre (Potato), et Tomate (Tomato).
    -  **Quelques images de dataset:**
     ![](/notebook/affichage_de_samples.png)

-   **Prétraitements & Pipeline de Données :**
    -   **Division des données :** Le jeu de données a été séparé en trois ensembles distincts avec une graine fixe pour la reproductibilité :
        -   **Entraînement :** 80%
        -   **Validation :** 10%
        -   **Test :** 10%
    -   **Redimensionnement :** Toutes les images ont été uniformisées à une taille de `256x256` pixels lors du chargement.
    -   **Normalisation :** Les valeurs des pixels de chaque image ont été mises à l'échelle dans l'intervalle `[0, 1]`.
    -   **Mise en lots (Batching) :** Les données sont groupées en lots de 32 pour optimiser l'entraînement et l'évaluation.
    -   **Data Augmentation :** Pour améliorer la généralisation du modèle et réduire le sur-apprentissage, des transformations aléatoires sont appliquées en temps réel **uniquement** sur les images du jeu d'entraînement. Celles-ci incluent :
        -   Retournements (`RandomFlip`)
        -   Rotations (`RandomRotation`)
        -   Zooms (`RandomZoom`)
        -   Ajustements de contraste (`RandomContrast`)
    


## 4. Modèle
-   **Architecture :** Réseau de Neurones Convolutifs (CNN) séquentiel, conçu pour l'extraction de caractéristiques et la classification. La structure est composée de plusieurs blocs, chacun optimisé pour une tâche spécifique :

    1.  **Blocs de Convolution (x3) :**
        -   Trois blocs successifs de `Conv2D` (avec 32, 64, puis 128 filtres) suivis par une `MaxPooling2D`.
        -   Ces blocs sont responsables de l'extraction des caractéristiques visuelles des images (bords, textures, formes).
        -   Chaque bloc est régularisé par une `BatchNormalization` pour stabiliser l'apprentissage et une couche `Dropout` pour réduire le sur-apprentissage.

    2.  **Bloc de Classification (Tête) :**
        -   Une couche `Flatten` pour transformer les cartes de caractéristiques 2D en un vecteur 1D.
        -   Deux couches `Dense` cachées (de 256 puis 128 neurones) qui agissent comme le "cerveau" du classifieur. Ces couches sont également régularisées avec `BatchNormalization` et `Dropout`.
        -   Une couche `Dense` finale avec 15 neurones et une activation `softmax`, qui produit la probabilité pour chacune des 15 classes de maladies.

    **Nombre total de paramètres :** Environ 29.6 millions. 

- **Framework :** TensorFlow 2.18 avec l'API de haut niveau Keras.  
- **Hyperparamètres & Stratégie d'Entraînement :**
    -   **Époques (`epochs`) :** Jusqu'à 36. L'entraînement peut s'arrêter plus tôt grâce au callback `EarlyStopping`.
    -   **Taille de lot (`batch size`) :** 32.
    -   **Optimiseur (`optimizer`) :** Adam.
    -   **Taux d'apprentissage (Learning Rate) :**
        -   **Initial :** `0.01` (Note : cette valeur est très élevée).
        -   **Dynamique :** Le taux d'apprentissage est automatiquement réduit pendant l'entraînement grâce au callback `ReduceLROnPlateau`.
    -   **Fonction de perte (`loss`) :** `SparseCategoricalCrossentropy`, car nos labels sont des entiers.
    -   **Callbacks :**
        -   `EarlyStopping`: Arrête l'entraînement si la perte de validation (`val_loss`) ne s'améliore pas pendant 5 époques, et restaure les meilleurs poids du modèle.
        -   `ReduceLROnPlateau`: Réduit le taux d'apprentissage d'un facteur de 10 si la `val_loss` ne s'améliore pas pendant 2 époques.


## 5. Résultats

- **Accuracy :** L'accuracy finale obtenue sur l'ensemble de test est de **86.24%**.
- **F1-Score :** Le F1-Score (pondéré) sur l'ensemble de test est de **0.86**.
- **Courbes d’entraînement :** Les courbes de `loss` et `accuracy` pour l'entraînement et la validation démontrent la bonne convergence du modèle sans sur-apprentissage majeur.  
  ![Courbes d'entraînement](/notebook/les_courbes_de_loss_et_accuracy.png)

- **Matrice de confusion :** La matrice de confusion sur l'ensemble de test permet d'analyser les erreurs de classification du modèle entre les différentes classes.  
  ![Matrice de confusion](/notebook/La_matrice_de_confusion.png)

## 6. Reproduction
### Environnement
- **Python :** 3.11+  
- **Bibliothèques principales :** (voir le fichier `requirements.txt`)
  ```
  tensorflow==2.10.0
  tensorflow-datasets==4.6.0
  matplotlib==3.5.3
  numpy==1.23.5
  seaborn==0.12.1
  ```

### Exécution
1. Cloner le dépôt :  
   `git clone https://github.com/votre-utilisateur/votre-projet.git`
2. Installer les dépendances :  
   `pip install -r requirements.txt`
3. Lancer le notebook principal qui contient le code pour le chargement, l'entraînement et l'évaluation :  
   `notebook/plantvillage-classification.ipynb`

## 7. Auteurs
- **Étudiant 1 :** Marouane
- **Étudiant 2 :** Imad

## 8. Licence
Le code de ce projet est distribué sous la **licence MIT**.  
Le jeu de données PlantVillage est disponible sous sa propre licence (généralement Creative Commons pour un usage non-commercial et de recherche).
