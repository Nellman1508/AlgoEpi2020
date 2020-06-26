'''
version 1.2
python version : 3.8
- Finds all csv files in the directory (Avoids all files with Merged in filename)
- add them togther while changing the name of their columns
- Save the obtained in a new Data Frame: Merged_Data.csv
bonus ignore unusual files
Merges Datum and Uhrzeit together and avoid replicates
Intended for use in the Project "Stromdaten-Analyse" for DatascienceI SS2020
at the Goethe University.
'''

from os import listdir, getcwd
import pandas as pd
from lib.preprocessing_powerplant_data import preprocessing

def data_frame_merger(dataframe1, dataframe2):
    if dataframe1 is None:
        return dataframe2
    merged_dataframe = dataframe1.merge(dataframe2,  on = "Datum und Uhrzeit", how = 'outer')
    #merged_dataframe.set_index("Datum und Uhrzeit", inplace = True)
    return merged_dataframe


def merger(path=0):
    if path == 0:
        print("Path missing for step Merging Powerplant Data.")

    folder_path = path + "\\" + "Kraftwerke" + "\\"
    preprocessing(folder_path, x=1)
    list_of_data_frames =  []
    one_data_frame = None
    broken_files = []


    for  csv_file in [file for x, file in enumerate(listdir(folder_path)) if file[-3:] == "csv"]:  # Browse in each data frame (csv data only)
            if csv_file.find("Merged") != -1: continue
            try:
                list_of_data_frames.append(pd.read_csv(folder_path + "\\" + csv_file, sep = ","))  # Save the data frame in a list
            except:
                broken_files.append(csv_file)
                print("Could not manage to convert,", csv_file)
                continue

            # Now me merge "Datum" und "Uhrzeit" togther and delete the "Uhrzeit"
            list_of_data_frames[-1]["Datum"] = \
            list_of_data_frames[-1]["Datum"] + " " + list_of_data_frames[-1]["Uhrzeit"]
            list_of_data_frames[-1].drop(columns=['Uhrzeit'], inplace = True)

            new_names = ["Datum und Uhrzeit"]
            for each_string in list(list_of_data_frames[-1].columns)[1:]:
                    new_names.append(each_string + csv_file[:-4])

            new_names_dict = dict( zip( list( list_of_data_frames[-1].columns ), new_names ))
            list_of_data_frames[-1].rename(columns = new_names_dict, inplace = True)  # OInput the name in the columns

    print('Merging data')
    while list_of_data_frames:
            one_data_frame = data_frame_merger(one_data_frame, list_of_data_frames[0])
            del[list_of_data_frames[0]]

    one_data_frame = one_data_frame.iloc[:2903,:]
    print('saving data')
    # Save the data
    one_data_frame.to_csv(path + "\\" + "workdir" + "\\" + "Merged_Data.csv")

    print('Done!')
