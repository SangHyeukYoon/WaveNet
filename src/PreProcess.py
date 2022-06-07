import random

import tensorflow as tf
import librosa
import soundfile as sf
import numpy as np

from os import listdir
from os.path import isfile, join

from MuLaw import muLaw, iMuLaw


class PreProcess:

    def __init__(self, filePath, audioLen):
        self.filePath = filePath
        self.audioLen = audioLen

        # Read speakers info.
        spkInfoPath = self.filePath + '\\speaker-info.txt'

        self.speakerNums = []

        with open(spkInfoPath, "r") as f:
            f.readline()
            table = f.readlines()

            for line in table:
                parse = line.split(" ")
                self.speakerNums.append(int(parse[0]))

        self.numSpk = len(self.speakerNums)

        # Make wav files path list.
        self.wavfilePath = []
        for num in self.speakerNums:
            spkFilePath = self.filePath + '\\wav48\\p' + str(num) + '\\'
            self.wavfilePath += [spkFilePath + path for path in listdir(spkFilePath) if isfile(join(spkFilePath, path))]

        self.numWav = len(self.wavfilePath)

        random.shuffle(self.wavfilePath)

    def getTrainData(self):
        duration = self.audioLen / 16000
        i = 0
        N = len(self.wavfilePath)

        while i < N:
            path = self.wavfilePath[i]
            i += 1

            # Check speaker number.
            parse = path.split('\\')
            spkNum = int(parse[-2][1:])
            c = tf.one_hot(self.speakerNums.index(spkNum), self.numSpk)
            c = tf.reshape(c, [1, -1, self.numSpk])

            # Read wav file.
            f, _ = librosa.load(path, sr=16000)
            f, _ = librosa.effects.trim(f)
            f = librosa.util.fix_length(f, size=self.audioLen)

            f = muLaw(f)
            f = tf.one_hot(f, 256)
            f = tf.expand_dims(f, axis=0)

            yield f, c

    def getSeed(self):
        path = self.wavfilePath[0]

        # Check speaker number.
        parse = path.split('\\')
        spkNum = int(parse[-2][1:])
        c = tf.one_hot(self.speakerNums.index(spkNum), self.numSpk)
        c = tf.reshape(c, [1, -1, self.numSpk])

        f, _ = librosa.load(path, sr=16000)
        f, _ = librosa.effects.trim(f)
        f = muLaw(f)
        f = tf.one_hot(f, 256)

        return f, c

    def getNumWav(self):
        return self.numWav

    def getNumSpk(self):
        return self.numSpk


if __name__ == '__main__':
    filePath = 'C:\\Users\\nyoon\\Documents\\VCTK-Corpus'
    preProcess = PreProcess(filePath, 16000 * 3)

    for f, c in preProcess.getTrainData():
        print(f.shape)

