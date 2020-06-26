# -*- coding: utf-8 -*-
"""
Main module of tje Project Stromdaten-Analyse.
Intended for use in the Project "Stromdaten-Analyse" for DatascienceI SS2020
at the Goethe University.
"""
from os import getcwd
from lib.data_frame_merger import merger
from lib.preprocessing_powerplant_data import preprocessing
from lib.KMeans_clustering import clustering
from lib.regression import regression


path = getcwd()
print("Currently Working in path: " + path)

# Step1: Merge the Powerplant Data
merger(path)

# Step 2: Preprocess all Data
pp_path = path + "\\" + "workdir" + "\\"
preprocessing(pp_path)

# Step 3: Clustering
clustering(path)

# Step 4: Regression
regression(path)