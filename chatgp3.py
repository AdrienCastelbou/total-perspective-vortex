import numpy as np

#Suppose que nous avons deux classes de données, "classe1" et "classe2", chacune contenant des époques de données EEG
#Suppose que les données sont organisées de la manière suivante :
#   data[classe][époque][channel]
#   où classe est 0 pour la première classe et 1 pour la seconde classe
#   époque est un entier correspondant à l'époque de données
#   channel est un entier correspondant au canal EEG

# Calcul des matrices de covariance pour chaque classe
cov_class1 = np.cov(data[0].T)
cov_class2 = np.cov(data[1].T)

# Calcul de la DVP pour chaque matrice de covariance
eig_vals_class, eig_vecs_class1 = np.linalg.eig(cov_class1)
eig_vals_class2, eig_vecs_class2 = np.linalg.eig(cov_class2)

# Sélectionner les vecteurs propres qui maximisent la différence de variance entre les classes
# Cela peut être fait en utilisant une métrique telle que le rapport de Rayleigh
# pour chaque vecteur propre, calculer le rapport de Rayleigh
rayleigh_ratios = []
for i in range(len(eig_vals_class1)):
    rayleigh_ratio = (eig_vals_class1[i] - eig_vals_class2[i]) / (eig_vals_class1[i] + eig_vals_class2[i])
    rayleigh_ratios.append(rayleigh_ratio)

# trier les vecteurs propres en fonction du rapport de Rayleigh
sorted_indices = np.argsort(rayleigh_ratios)[::-1]

# Sélectionner les vecteurs propres les plus importants en fonction du nombre de caractéristiques souhaitées
important_vecs_class1 = eig_vecs_class1[:, sorted_indices[:num_features]]
important_vecs_class2 = eig_vecs_class2[:, sorted_indices[:num_features]]

# Appliquer les vecteurs propres sélectionnés aux données pour obtenir les caractéristiques CSP
csp_features_class1 = np.dot(important_vecs_class1.T, data[0])
csp_features_class2 = np.dot(important_vecs_class2.T, data[1])

