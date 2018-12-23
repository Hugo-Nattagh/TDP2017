from vad import VoiceActivityDetector
from keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wave
import os
import librosa
import librosa.display
import pandas as pd
from pydub import AudioSegment
import tkinter as tk
from tkinter import Canvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import datetime

# GETTING FILE

FILE_FOLDER = 'Data/'
FILE_NAME = 'Le Grand Débat (TF1)'  # Don't type the file extension (only wav files are supported)
fullFileName = FILE_FOLDER + FILE_NAME + '.wav'

# Get file
waveFile = wave.open(fullFileName, 'r')

# If file has 2 channels (Stereo), this creates a copy of file with 1 channel (mono),
# and import that file instead
if waveFile.getnchannels() == 2:
    print('wav file is stereo\nCreating a mono wav file')
    x = AudioSegment.from_wav(fullFileName)
    x = x.set_channels(1)
    x.export(FILE_FOLDER + FILE_NAME + '-mono.wav', format="wav")
    waveFile = wave.open(FILE_FOLDER + FILE_NAME + '-mono.wav', 'r')

# Getting some variables from file
signal = waveFile.readframes(-1)
signal = np.fromstring(signal, 'Int16')
frameRate = waveFile.getframerate()
Time = np.linspace(0, len(signal) / frameRate, num=len(signal))
totalFrames = waveFile.getnframes()
wavFileLengthSec = totalFrames / frameRate

#  VOICE ACTIVITY DETECTION (VAD)

vad = VoiceActivityDetector(fullFileName)
raw_detection = vad.detect_speech()
speech_labels, starters, enders = vad.convert_windows_to_readible_labels(raw_detection)

# list of beginnings of voice detected segments (in frames)
starterFrames = [int(i) for i in starters]
# list of endings of voice detected segments (in frames)
enderFrames = [int(i) for i in enders]

# if the last voice detected segment isn't closed, use the last frame as end of this segment
if len(starterFrames) > len(enderFrames):
    enderFrames.append(totalFrames)

# list of beginnings of voice detected segments (in milliseconds)
starterMs = [int((i / frameRate) * 1000) for i in starterFrames]
# list of endings of voice detected segments (in milliseconds)
enderMs = [int((i / frameRate) * 1000) for i in enderFrames]

# EXPORTING TEMPORARY VOICE SEGMENTS

audioFile = AudioSegment.from_wav(fullFileName)

lengthSecList = []

for i in range(len(starterMs)):
    tempWav = audioFile[starterMs[i]:enderMs[i]]
    lengthSecList.append((enderMs[i] - starterMs[i]) / 1000)
    tempWav.export("temp/tempWav-" + str(i) + ".wav", format="wav")

# CLASSIFYING

# import model
model = load_model('Model/model.h5')

# list of temp files
tempDir = os.listdir('temp/')
# number of temp files
nb_files = len(tempDir)

# creating the dataframe containing the sequences to be classified
colList = ['Seq', 'Name', 'Length']
d = {}

for i in colList:
    d[i] = range(nb_files)

df = pd.DataFrame(d, columns=colList).astype('object')

for i in range(len(lengthSecList)):
    df.iloc[i, 2] = lengthSecList[i]

z = 0
for file in tempDir:
    data, sampling_rate = librosa.load('temp/' + file, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    df.iloc[z, 0] = mfccs
    df.iloc[z, 1] = None
    os.remove('temp/' + file)
    z += 1

X = np.array(df.Seq.tolist())
predictions = model.predict(X)
classesNum = [np.argmax(y, axis=None, out=None) for y in predictions]

for i in range(len(classesNum)):
    if classesNum[i] == 0:
        df.iloc[i, 1] = 'Hamon'
    elif classesNum[i] == 1:
        df.iloc[i, 1] = 'MLP'
    elif classesNum[i] == 2:
        df.iloc[i, 1] = 'JLM'
    elif classesNum[i] == 3:
        df.iloc[i, 1] = 'Macron'
    elif classesNum[i] == 4:
        df.iloc[i, 1] = 'Fillon'

dfLengthSec = df.groupby('Name').Length.sum().sort_values(ascending=False).to_frame()

totalTalkTime = df.Length.sum()

dfLengthSec.reset_index(level=0, inplace=True)
dfLengthSec['Percentage_Talk'] = round((dfLengthSec['Length'] / totalTalkTime) * 100, 1)


lengthStringDic = {}
percentStringDic = {}
percentIntDic = {}

for i in range(len(dfLengthSec)):
    name = dfLengthSec.Name[i]
    minutes = int(dfLengthSec.Length[i]) // 60
    seconds = int(dfLengthSec.Length[i]) % 60
    percentInt = int(dfLengthSec.Percentage_Talk[i])
    if minutes == 0:
        lengthStringDic[name] = str(seconds) + ' secondes'
    else:
        lengthStringDic[name] = str(minutes) + ' minutes ' + str(seconds) + ' secondes'
    percentIntDic[name] = percentInt
    percentStringDic[name] = str(dfLengthSec.Percentage_Talk[i]) + ' %'

print(dfLengthSec)
print(lengthStringDic)

# PLOTTING

fig = plt.figure(1, figsize=(15, 3))
plt.title(FILE_NAME)

axes = plt.gca()


def timeTicks(x, pos):
    d = datetime.timedelta(seconds=x)
    return str(d)


formatter = matplotlib.ticker.FuncFormatter(timeTicks)
axes.xaxis.set_major_formatter(formatter)

if starterFrames[0] != 0:
    plt.plot(Time[:starterFrames[i]], signal[:starterFrames[i]], color='black')

if enderFrames[-1] != totalFrames:
    plt.plot(Time[enderFrames[-1]:], signal[enderFrames[-1]:], color='black')

for i in range(len(starterFrames)):
    if df.iloc[i, 1] == 'Hamon':
        plt.plot(Time[starterFrames[i]:enderFrames[i]], signal[starterFrames[i]:enderFrames[i]], color='#ff0000')
    elif df.iloc[i, 1] == 'JLM':
        plt.plot(Time[starterFrames[i]:enderFrames[i]], signal[starterFrames[i]:enderFrames[i]], color='#990000')
    elif df.iloc[i, 1] == 'MLP':
        plt.plot(Time[starterFrames[i]:enderFrames[i]], signal[starterFrames[i]:enderFrames[i]], color='#002699')
    elif df.iloc[i, 1] == 'Macron':
        plt.plot(Time[starterFrames[i]:enderFrames[i]], signal[starterFrames[i]:enderFrames[i]], color='#6600cc')
    elif df.iloc[i, 1] == 'Fillon':
        plt.plot(Time[starterFrames[i]:enderFrames[i]], signal[starterFrames[i]:enderFrames[i]], color='#0040ff')
    if i < len(starterFrames) - 1:
        plt.plot(Time[enderFrames[i]:starterFrames[i + 1]], signal[enderFrames[i]:starterFrames[i + 1]], color='black')

plt.close()

# GUI

window = tk.Tk()
window.title('Results')

#  Plot (Frame0)
plotFrame = tk.Frame(window, bd=4, relief="ridge")
plotFrame.grid(column=0, row=0, columnspan=2)

canvas = FigureCanvasTkAgg(fig, master=plotFrame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, plotFrame)
toolbar.update()
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Pictures, Time Length and percentage
# MACRON (Frame1)___________________________________________
Frame1 = tk.Frame(window, bd=4, relief="ridge")
Frame1.grid(column=0, row=1)

photo = tk.PhotoImage(file="resources/lemac.png")
v = tk.Label(Frame1, image=photo)
v.photo = photo
v.grid(column=0, row=0, rowspan=2)

if 'Macron' in str(dfLengthSec.Name):
    labelLength1 = tk.Label(Frame1, text=lengthStringDic['Macron'])
    labelLength1.configure(font=("Times New Roman", 12, "bold"))
    if len(dfLengthSec) > 1:
        labelLength1.grid(column=1, row=0, columnspan=2)
        canvas1 = Canvas(Frame1, width=percentIntDic['Macron'] + 5, height=20)
        canvas1.grid(column=1, row=1)
        a = canvas1.create_rectangle(0, 0, percentIntDic['Macron'], 20, fill='#6600cc')
        labelLength11 = tk.Label(Frame1, text=percentStringDic['Macron'])
        labelLength11.configure(font=("Times New Roman", 12, "bold"))
        labelLength11.grid(column=2, row=1)
    else:
        labelLength1.grid(column=1, row=0, rowspan=2)

# HAMON (Frame2)___________________________________________
Frame2 = tk.Frame(window, bd=4, relief="ridge")
Frame2.grid(column=0, row=2)

photo = tk.PhotoImage(file="resources/hamham.png")
w = tk.Label(Frame2, image=photo)
w.photo = photo
w.grid(column=0, row=0, rowspan=2)

if 'Hamon' in str(dfLengthSec.Name):
    labelLength2 = tk.Label(Frame2, text=lengthStringDic['Hamon'])
    labelLength2.configure(font=("Times New Roman", 12, "bold"))
    if len(dfLengthSec) > 1:
        labelLength2.grid(column=1, row=0, columnspan=2)
        canvas2 = Canvas(Frame2, width=percentIntDic['Hamon'] + 5, height=20)
        canvas2.grid(column=1, row=1)
        a = canvas2.create_rectangle(0, 0, percentIntDic['Hamon'], 20, fill='#ff0000')
        labelLength21 = tk.Label(Frame2, text=percentStringDic['Hamon'])
        labelLength21.configure(font=("Times New Roman", 12, "bold"))
        labelLength21.grid(column=2, row=1)
    else:
        labelLength2.grid(column=1, row=0, rowspan=2)

# FILLON (Frame3)___________________________________________
Frame3 = tk.Frame(window, bd=4, relief="ridge")
Frame3.grid(column=0, row=3)

photo = tk.PhotoImage(file="resources/filloss.png")
vv = tk.Label(Frame3, image=photo)
vv.photo = photo
vv.grid(column=0, row=0, rowspan=2)

if 'Fillon' in str(dfLengthSec.Name):
    labelLength3 = tk.Label(Frame3, text=lengthStringDic['Fillon'])
    labelLength3.configure(font=("Times New Roman", 12, "bold"))
    if len(dfLengthSec) > 1:
        labelLength3.grid(column=1, row=0, columnspan=2)
        canvas3 = Canvas(Frame3, width=percentIntDic['Fillon'] + 5, height=20)
        canvas3.grid(column=1, row=1)
        a = canvas3.create_rectangle(0, 0, percentIntDic['Fillon'], 20, fill='#0040ff')
        labelLength31 = tk.Label(Frame3, text=percentStringDic['Fillon'])
        labelLength31.configure(font=("Times New Roman", 12, "bold"))
        labelLength31.grid(column=2, row=1)
    else:
        labelLength3.grid(column=1, row=0, rowspan=2)

# LE PEN (Frame4)___________________________________________
Frame4 = tk.Frame(window, bd=4, relief="ridge")
Frame4.grid(column=1, row=1)

photo = tk.PhotoImage(file="resources/lepenpenpen.png")
ww = tk.Label(Frame4, image=photo)
ww.photo = photo
ww.grid(column=0, row=0, rowspan=2)

print(dfLengthSec.Length.loc[df['Name'] == 'MLP'])

if 'MLP' in str(dfLengthSec.Name):
    labelLength4 = tk.Label(Frame4, text=lengthStringDic['MLP'])
    labelLength4.configure(font=("Times New Roman", 12, "bold"))
    if len(dfLengthSec) > 1:
        labelLength4.grid(column=1, row=0, columnspan=2)
        canvas4 = Canvas(Frame4, width=percentIntDic['MLP'] + 5, height=20)
        canvas4.grid(column=1, row=1)
        a = canvas4.create_rectangle(0, 0, percentIntDic['MLP'], 20, fill='#002699')
        labelLength41 = tk.Label(Frame4, text=percentStringDic['MLP'])
        labelLength41.configure(font=("Times New Roman", 12, "bold"))
        labelLength41.grid(column=2, row=1)
    else:
        labelLength4.grid(column=1, row=0, rowspan=2)

# MÉLENCHON (Frame5)___________________________________________
Frame5 = tk.Frame(window, bd=4, relief="ridge")
Frame5.grid(column=1, row=2)

photo = tk.PhotoImage(file="resources/grosméluche.png")
vw = tk.Label(Frame5, image=photo)
vw.photo = photo
vw.grid(column=0, row=0, rowspan=2)

if 'JLM' in str(dfLengthSec.Name):
    labelLength5 = tk.Label(Frame5, text=lengthStringDic['JLM'])
    labelLength5.configure(font=("Times New Roman", 12, "bold"))
    if len(dfLengthSec) > 1:
        labelLength5.grid(column=1, row=0, columnspan=2)
        canvas5 = Canvas(Frame5, width=percentIntDic['JLM'] + 5, height=20)
        canvas5.grid(column=1, row=1)
        a = canvas5.create_rectangle(0, 0, percentIntDic['JLM'], 20, fill='#990000')
        labelLength51 = tk.Label(Frame5, text=percentStringDic['JLM'])
        labelLength51.configure(font=("Times New Roman", 12, "bold"))
        labelLength51.grid(column=2, row=1)
    else:
        labelLength5.grid(column=1, row=0, rowspan=2)

window.mainloop()
