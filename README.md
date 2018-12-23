# TDP2017

### Vocal Recognition on Politicians' Alloted Time to Speak

![](https://raw.githubusercontent.com/Hugo-Nattagh/TDP2017/master/resources/GUI.png)

I wanted to measure the actual alloted time for the main 5 candidates running for president in 2017 in France. (In an audio file)


#### What's going on

In the `nn.py` script, I extract the MFCCs (with Librosa) out of the audio segments I previously selected.

I train my neural networks with those MFCCs.

In the `tdp2017.py` script, I import the audio file on which I want to the predictions.

I use the [vad](https://github.com/marsbroshok/VAD-python) (by [Alexander Usoltsev](https://github.com/marsbroshok)) package to extract audio chunks when some voice acitvity is detected.

I extract the MFCCs from those chunks.

The model displays the predictions and some details in a tkinter GUI, with a matplotlib plot integrated.

#### Why

I made this to assist me in my job for the CSA (Superior Council of the Audiovisual). 

Not only it would save me some time, but it could potentially get better at recognizing the voice of a candidate than a human.

#### Files Description

- `name_changer.py`: Script to rename audio segment files, to be trained
- `nn.py`: Script to create and train the neural network
- `vad.py`: Script to detect voice activity
- `tdp2017.py`: Script to predict
- `model.h5`: Neural network saved