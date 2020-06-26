"""
Intended for use in the Project "Stromdaten-Analyse" for DatascienceI SS2020
at the Goethe University.
"""
from os import listdir
import re


def preprocessing(path=0, x=0):
    if path == 0:
        print("Path missing for step preprocessing.")


    folder_path = path
    print("Preprocessing Data in Path: " + folder_path)
    for  csv_file in [file for x, file in enumerate(listdir(folder_path)) \
                      if file[-3:] == "csv"]:
        mylist = []
        if "pp_" in csv_file or "Merged_" in csv_file:
            continue
        csv_path = folder_path + csv_file
        with open(csv_path, 'r') as my_file:

            list_of_lines = my_file.readlines()
            for one_line in list_of_lines:
                if "Physikalischer" not in csv_file:
                    one_line = one_line.replace("-", "0")
                one_line = one_line.replace("ä", "_")
                one_line = one_line.replace("ö", "_")
                one_line = one_line.replace("ü", "_")
                one_line = one_line.replace(";", ",")
                one_line = one_line[:10] + one_line[10:].replace(".", "")
                mylist.append(one_line)

        if x == 0:
            # Regex Tester: https://regex101.com/ 24.06.2020
            csv_file = re.sub("_\d+.*", ".csv", csv_file)
            new_path = folder_path + "pp_" + csv_file
        else:
            new_path = folder_path + csv_file

        with open(new_path, 'w') as my_file:
            for one_line in mylist:
                my_file.write(one_line)
    print("Done Preprocessing")
