# -*- coding: utf-8 -*-
"""
Module to create linear and polynomial regression 3 different files.
Intended for use in the Project "Stromdaten-Analyse" for DatascienceI SS2020
at the Goethe University.
"""

from os import getcwd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lib.min2h import min_to_h
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import seaborn as seabornInstance


def regression(path):
    print("Start Regression")
    path_1 = path + "\\" + "workdir" + "\\" + "Merged_Data.csv"

    path_2 = path + "\\" + "workdir" + "\\" + "pp_Realisierte_Erzeugung.csv"

    path_3 = path + "\\" + "workdir" + "\\" + "pp_Realisierter_Stromverbrauch.csv"

    path_4 = path + "\\" + "workdir" + "\\" + "pp_Physikalischer_Stromfluss.csv"

    powerplant_data = pd.read_csv(path_1)
    generated_power = min_to_h(pd.read_csv(path_2))
    consumed_power = min_to_h(pd.read_csv((path_3)))
    export_import = pd.read_csv(path_4)


    powerplant_data["Total"] = powerplant_data.sum(axis=1)
    # generated_power = generated_power.drop("Datum", "Uhrzeit", axis=1)
    generated_power["Total"] = generated_power.iloc[2:].sum(axis=1)
    export_import = export_import.drop(export_import.columns[3:], axis=1)
    export_import = export_import.drop(export_import.columns[0:2], axis=1)
    date = powerplant_data["Datum und Uhrzeit"].tolist()

    # X-Axis ticks and labels
    x_positions = [0, round((len(date))/2), (len(date)-1)]
    x_label = [date[x_positions[0]], date[x_positions[1]], date[x_positions[2]]]


    df = pd.DataFrame(columns=["Hours", "produced_total", "given_p_total", "consumed_total", "Nettoexport"])
    df["produced_total"] = powerplant_data["Total"]
    df["given_p_total"] = generated_power["Total"]
    df["consumed_total"] = consumed_power["Gesamt[MWh]"]
    df["Nettoexport"] = export_import["Physikalischer Nettoexport[MWh]"]
    df["Hours"] = df.index


    csv_path = path + "\\" + "outdir" + "\\" + "Merged_production_consumption.csv"
    df.to_csv(csv_path)
    df = df.drop(["Nettoexport"], axis=1)
    df = df.fillna(0).astype(float)

    # Figure 1
    x = df["Hours"]
    y1 = df["produced_total"]
    y2 = df["given_p_total"]
    y3 = df["consumed_total"]
    fig_1, ax1 = plt.subplots()
    fig_1.set_size_inches(12, 7, forward=True)
    ax1.plot(x, y1, label='calculated_produced_total')
    ax1.plot(x, y2, label='SMARD_produced_total')
    ax1.plot(x, y3, label='consumed_total')
    ax1.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1, fontsize = "x-small")
    plt.xlabel('Time')
    plt.ylabel('MW')
    positions = x_positions
    labels = x_label
    plt.xticks(positions, labels)
    plt.title("Powerproduction and Consumption")
    print("saving first plot in", path + "\\outdir\\" + "Stromerzeugung_und_Verbrauch.png")
    plt.savefig(path + "\\outdir\\" + "Stromerzeugung_und_Verbrauch.png")
    plt.show()

    # print(df)
    # Figure 2
    df.plot(x = "Hours" , y = 'produced_total', kind = 'line', figsize=(12,7))
    plt.title('Calculated Produced Power')
    plt.xlabel('Time')
    plt.ylabel('MW')
    positions = x_positions
    labels = x_label
    plt.xticks(positions, labels)
    plt.show()

    # plt.figure(figsize=(15,10))
    # plt.tight_layout()
    # seabornInstance.distplot(df['produced_total'])


    # linear regression
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 26.06.2020
    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html 26.26.2020
    # https://realpython.com/linear-regression-in-python/#polynomial-regression 24.06.2020
    """Regression for our own produced_total"""
    X = df['Hours'].values.reshape(-1,1)
    y = df['produced_total'].values.reshape(-1,1)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train1, y_train1)


    y_pred1 = regressor.predict(X_test1)
    my_dataframe = pd.DataFrame({'Actual': y_test1.flatten(), 'Predicted': y_pred1.flatten()})
    my_dataframe

    # my_dataframe1 = my_dataframe.head(25)
    # my_dataframe1.plot(kind='bar',figsize=(16,10))
    # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # plt.show()
    '''
    ##### Fig 3
    plt.scatter(X_test1, y_test1,  color='gray')
    plt.plot(X_test1, y_pred1, color='red', linewidth=2)
    plt.show()
    '''
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test1, y_pred1))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test1, y_pred1))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test1, y_pred1)))




    """Regression for the given_p_total we downloaded."""
    X = df['Hours'].values.reshape(-1,1)
    y = df['given_p_total'].values.reshape(-1,1)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train2, y_train2)


    y_pred2 = regressor.predict(X_test2)
    my_dataframe = pd.DataFrame({'Actual': y_test2.flatten(), 'Predicted': y_pred2.flatten()})
    my_dataframe

    # my_dataframe1 = my_dataframe.head(25)
    # my_dataframe1.plot(kind='bar',figsize=(16,10))
    # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # plt.show()
    '''
    ##### Fig 4
    plt.scatter(X_test2, y_test2,  color='gray')
    plt.plot(X_test2, y_pred2, color='red', linewidth=2)
    plt.show()
    '''
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test2, y_pred2))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test2, y_pred2))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test2, y_pred2)))




    """Regression for the consumed_total we downloaded."""
    X = df['Hours'].values.reshape(-1,1)
    y = df['consumed_total'].values.reshape(-1,1)

    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train3, y_train3)


    y_pred3 = regressor.predict(X_test3)
    my_dataframe = pd.DataFrame({'Actual': y_test3.flatten(), 'Predicted': y_pred3.flatten()})
    my_dataframe

    # my_dataframe1 = my_dataframe.head(25)
    # my_dataframe1.plot(kind='bar',figsize=(16,10))
    # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # plt.show()
    ##### Fig 5
    '''
    plt.scatter(X_test3, y_test3,  color='gray')
    plt.plot(X_test3, y_pred3, color='red', linewidth=2)
    plt.show()
    '''

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test3, y_pred3))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test3, y_pred3))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test3, y_pred3)))

    ######## Ultimate Figure 1

    my_figure1 = plt.figure(1, figsize = (20, 5))

    sub_figure_1 = my_figure1.add_subplot(131)  ##############
    ###### Fig3 ##########
    """Regression for our own produced_total"""
    X = df['Hours'].values.reshape(-1,1)
    y = df['produced_total'].values.reshape(-1,1)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train1, y_train1)


    y_pred1 = regressor.predict(X_test1)
    my_dataframe = pd.DataFrame({'Actual': y_test1.flatten(), 'Predicted': y_pred1.flatten()})
    my_dataframe

    ##### Fig 3
    plt.scatter(X_test1, y_test1,  color="DarkRed", label='calculated_produced_total')
    plt.plot(X_test1, y_pred1, color='red', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('MW')
    positions = x_positions
    labels = x_label
    plt.xticks(positions, labels)
    plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1)


    sub_figure_2 = my_figure1.add_subplot(132)  ##########
    ###### Fig4 ##########
    """Regression for the given_p_total we downloaded."""
    X = df['Hours'].values.reshape(-1,1)
    y = df['given_p_total'].values.reshape(-1,1)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train2, y_train2)


    y_pred2 = regressor.predict(X_test2)
    my_dataframe = pd.DataFrame({'Actual': y_test2.flatten(), 'Predicted': y_pred2.flatten()})
    my_dataframe


    ##### Fig 4
    plt.scatter(X_test2, y_test2,  color='DarkGreen', label='SMARD_produced_total')
    plt.plot(X_test2, y_pred2, color='chartreuse', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.title("Linear Regression")
    positions = x_positions
    labels = x_label
    plt.xticks(positions, labels)
    plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1)


    sub_figure_2 = my_figure1.add_subplot(133)  #############
    ###### Fig5 ##########
    """Regression for the consumed_total we downloaded."""
    X = df['Hours'].values.reshape(-1,1)
    y = df['consumed_total'].values.reshape(-1,1)

    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train3, y_train3)

    y_pred3 = regressor.predict(X_test3)
    my_dataframe = pd.DataFrame({'Actual': y_test3.flatten(), 'Predicted': y_pred3.flatten()})
    my_dataframe


    ##### Fig 5
    plt.scatter(X_test3, y_test3,  color='Blue', label='SMARD_consumed_total')
    plt.plot(X_test3, y_pred3, color='deepskyblue', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('MW')
    positions = x_positions
    labels = x_label
    plt.xticks(positions, labels)
    plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1)


    print("saving 2nd plot in", path + "\\outdir\\" + "linear_regression.png")
    plt.savefig(path + "\\outdir\\" + "linear_regression.png")
    plt.show()


    ##### Fig 6
    """Print all linear Regression Models in one figure"""

    plt.figure(figsize=(12,7))
    plt.scatter(X_test1, y_test1, color="DarkRed", label = 'calculated_produced_total')
    plt.scatter(X_test2, y_test2, color="DarkGreen", label = 'SMARD_produced_total')
    plt.scatter(X_test3, y_test3, color="Blue", label = 'SMARD_consumed_total')
    plt.plot(X_test1, y_pred1, linewidth = 2, color = "Red")
    plt.plot(X_test2, y_pred2, linewidth = 2, color = "chartreuse")
    plt.plot(X_test3, y_pred3, linewidth = 2, color = "deepskyblue")
    plt.title('Linear Regression')
    plt.xlabel('Time')
    plt.ylabel('MW')
    positions = x_positions
    labels = x_label
    plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1)
    plt.xticks(positions, labels)

    plt.show()

    # polynomial regression
    # https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/ 26.26.2020
    print("Polynom Regression")
    X = df['Hours'].values.reshape(-1,1)
    y = df['produced_total'].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    '''
    ##### Fig 7
    plt.scatter(X, y, color='pink', marker="+")
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title("Poly Reg")
    plt.xlabel('Hours')
    plt.ylabel('Strom')
    plt.show()
    pol_reg.predict(poly_reg.fit_transform([[5.5]]))



    X = df['Hours'].values.reshape(-1,1)
    y = df['given_p_total'].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)

    ##### Fig 8
    plt.scatter(X, y, color='pink', marker="+")
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title("Poly Reg")
    plt.xlabel('Hours')
    plt.ylabel('Strom')
    plt.show()
    pol_reg.predict(poly_reg.fit_transform([[5.5]]))




    X = df['Hours'].values.reshape(-1,1)
    y = df['consumed_total'].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    ##### Fig 9
    plt.scatter(X, y, color='pink', marker="+")
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title("Poly Reg")
    plt.xlabel('Hours')
    plt.ylabel('Strom')
    plt.show()
    pol_reg.predict(poly_reg.fit_transform([[5.5]]))
    '''
    #### Ultimate Figure 2

    my_figure = plt.figure(1, figsize = (20, 7))

    sub_figure_1 = my_figure.add_subplot(131)
    ##### Fig 7
    X = df['Hours'].values.reshape(-1,1)
    y = df['produced_total'].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    ##### Fig 7
    plt.scatter(X, y, color='firebrick', marker="+", label='calculated_produced_total')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.ylim(ymin=0)
    positions = x_positions
    labels = x_label
    plt.xticks(positions, labels)
    plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1)
    pol_reg.predict(poly_reg.fit_transform([[5.5]]))

    sub_figure_2 = my_figure.add_subplot(132)
    ##### Fig 8
    X = df['Hours'].values.reshape(-1,1)
    y = df['given_p_total'].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)

    ##### Fig 8
    plt.scatter(X, y, color='orangered', marker="+", label='SMARD_produced_total')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title("Polynomial Regression")
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.ylim(ymin=0)
    positions = x_positions
    labels = x_label
    plt.xticks(positions, labels)
    plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1)
    pol_reg.predict(poly_reg.fit_transform([[5.5]]))

    sub_figure_2 = my_figure.add_subplot(133)

    ##### Fig 9
    X = df['Hours'].values.reshape(-1,1)
    y = df['consumed_total'].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    ##### Fig 9
    plt.scatter(X, y, color='chocolate', marker="+", label='SMARD_consumed_total')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.ylim(ymin=0)
    positions = x_positions
    labels = x_label
    plt.xticks(positions, labels)
    plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1)
    #sub_figure_1.plot()
    #sub_figure_2.plot()

    print("saving 3rd plot in", path + "\\outdir\\" + "polynomial_regression.png")
    plt.savefig(path + "\\outdir\\" + "polynomial_regression.png")
    plt.show()
    print("Finished Regression")
