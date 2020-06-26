# -*- coding: utf-8 -*-
"""
Intended for use in the Project "Stromdaten-Analyse" for DatascienceI SS2020
at the Goethe University.
"""


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from os import getcwd
from sklearn.cluster import KMeans
import seaborn as sns
import re
from copy import deepcopy


def clustering(path=0):
    if path == 0:
        print("Path missing for step clustering.")

    print("Clustering")

    path_1 = path + "\\" + "workdir" + "\\Merged_Data.csv"
    path_2 = path + "\\" + "workdir" + "\\Kraftwerkstypen.csv"

    energy_data = pd.read_csv(path_1, index_col=0)
    plant_type = pd.read_csv(path_2, index_col=0)

    energy_data = energy_data[energy_data.columns]
    power_plants = list(energy_data.columns)

    for i in range(0,len(power_plants)):  # Change the names
        power_plants[i] = re.sub(r".*MW.", "" , power_plants[i])
    energy_data.columns = power_plants


    # This finds the columns with the same name and sums their values
    for i in range( len(list(energy_data.columns))-1, 0, -1):
        if energy_data.columns[i] == energy_data.columns[i - 1]:
            energy_data[energy_data.columns[i-1]] = energy_data[energy_data.columns[i]] + energy_data[energy_data.columns[i -1 ]]
            #energy_data = energy_data.drop(energy_data.columns[i], axis = 1)
            # this is to delete the uses Column
            energy_data = pd.concat([energy_data.iloc[:, :i], energy_data.iloc[:, i+1:]], axis = 1 )

    csv_path_2 =  path + "\\" + "workdir" + "\\Kraftwerkstypen.csv"

    my_data = energy_data

    my_data.drop( columns = "Datum und Uhrzeit", inplace = True)  # Delete the time Colum
    my_new_dataframe = my_data.sum().to_frame()  # Create a new data frame containg the sum of all coumns
    my_new_dataframe.rename(columns={0: 'Total'}, inplace = True)  # Rename

    my_data_types = pd.read_csv(csv_path_2, index_col = 0)

    #my_new_dataframe = my_new_dataframe.join(my_data_types)
    #xxx = my_new_dataframe.merge(my_data_types, on = my_new_dataframe.index , left_index=True, right_index=True)
    my_merged_data = pd.concat([my_new_dataframe, my_data_types], axis = 1, sort= False)


    my_merged_data.to_csv( path +"\\outdir\\"+ "ClusterFrame.csv")

    df = my_merged_data
    df = df.fillna(0)
    data_numbers = np.array(df)

    data_numbers_2 = deepcopy(data_numbers)

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 26.06.2020
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html 26.06.2020
    preprocessd_data = preprocessing.StandardScaler().fit(data_numbers).transform(data_numbers.astype(float))
    km = KMeans(n_clusters=3, algorithm="full")
    km.fit(preprocessd_data)
    clusters = km.labels_.tolist()

    df["Type"] = df.drop(["Total"], axis = 1).idxmax(axis=1)
    df = df.drop(["fossile_fuel", "nuclear", "renewable", "storage"], axis = 1)
    #my_data["Type"] = my_data.idxmax(axis=1)
    df.insert(len(df.columns), "Cluster", clusters)

    df = df.sort_values("Cluster")

    indices = list(df.index)
    gen_method = list(df["Type"].tolist())
    for i in range(0, len(indices)):
        indices[i] = indices[i].replace(r"_20.*", " ") + gen_method[i]
    df.index = indices
    df = df.drop(["Type"], axis=1)

    # ax = df.plot(kind="line")
    # df.plot(secondary_y=True, ax=ax)
    # plt.show

    df["Cluster"] = clusters

    print("Clustering finished")
