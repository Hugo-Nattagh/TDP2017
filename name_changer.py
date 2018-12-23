import os

NAME = 'Macron'

NB_FILES_ALREADY_SET = 20

directory = 'Data/' + NAME + '/'

files_list = os.listdir(directory)

toReplace = directory + 'REPLAY - Discours dEmmanuel Macron au sommet de la Francophonie-'

replacement = directory + NAME + '-'

for i in range(1, len(files_list)):
    os.rename(toReplace + str(i) + '.wav', replacement + str(i + NB_FILES_ALREADY_SET) + '.wav')
