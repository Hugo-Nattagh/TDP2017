import os

directory = 'Data/Macron/'

files_list = os.listdir(directory)

toReplace = directory + 'REPLAY - Discours dEmmanuel Macron au sommet de la Francophonie-'

replacement = directory + 'Macron-'

for i in range(1, len(files_list)):
    os.rename(toReplace + str(i) + '.wav', replacement + str(i + 907) + '.wav')
