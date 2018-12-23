import librosa
import librosa.display
import os
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils

nameList = ['Hamon', 'MLP', 'JLM', 'Macron', 'Fillon']
modelFolder = 'Model/'


def get_df(name_list):

    df_final = pd.DataFrame(columns=['Seq', 'Name'])
    for name in name_list:
        nb_files = len(os.listdir('Data/' + name + '/'))
        colList = ['Seq', 'Name']
        d = {}
        for i in colList:
            d[i] = range(nb_files)
        df = pd.DataFrame(d, columns=colList)
        df = df.astype('object')
        df = extract(name, df)
        df_final = pd.concat([df_final, df], axis=0)
    return df_final


def extract(name, df):
    directory = 'Data/' + name + '/'
    dirList = os.listdir(directory)
    z = 0
    for file in dirList:
        data, sampling_rate = librosa.load(directory + file, res_type='kaiser_fast')
        print(file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        df.iloc[z, 0] = mfccs
        df.iloc[z, 1] = name
        z += 1
    return df


df = get_df(nameList)

X = np.array(df.Seq.tolist())
y = np.array(df.Name.tolist())

lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))

# print('Target (names) y:')
# print(y)
# yc = [np.argmax(x, axis=None, out=None) for x in y]
# print('Target (class) y:')
# print(yc)

num_labels = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(X_train, y_train, batch_size=100, epochs=500)

predictions = model.predict(X_test)

print('\nAccuracy:\nRoot mean squared error: %.2f \nR² score: %.2f\n' % (np.sqrt(mean_squared_error(y_test, predictions)), r2_score(y_test, predictions)))

# print(predictions)
# print('___________SEP__________')
# print(y_test)

model.save(modelFolder + 'model.h5')
