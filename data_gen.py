#!/usr/bin/env python
import numpy as np
import os


# parameters
cycleLength = 24  # cycleLength is D -- we consider features in first D days
sexRisks = {
    "unprotected_sex": 0.5,
    "protected_sex": 0.2,
    "withdrawl_sex": 0.3,
}
emotionImpacts = {
    "emotion_happy": 1.2,
    "emotion_neutral": 1.0,
    "emotion_sad": 0.8,
}

symptomNames = sexRisks.keys() + emotionImpacts.keys()
symptomIndices = {name: idx for idx, name in enumerate(symptomNames)}


# TODO: use some better way to simulate this instead of fixed fertilities
def genFertilities(cycleLength):
    fertilities = np.array([
        1, 1, 1, 1, 1,
        1, 2, 3, 10, 12,
        13, 11, 10, 8, 10,
        9, 5, 3, 2, 1,
        1, 1, 1, 1, 1,
        2, 1, 1, 2, 1,
        1, 1, 0, 0, 0,
    ], dtype=float)
    maxFertility = 0.8
    assert cycleLength <= fertilities.shape[0]
    return fertilities[:cycleLength] / np.max(fertilities) * maxFertility


def flipWithProb(prob):
    return np.random.random() <= prob


# Basic assumptions for single cycle data generation
#   - fertility: agrees with fertility window. Some factors like age, emotion
#     can affect fertility on the cycle or on a specific day.
#   - sex: different sex types have different risks, and it should be
#     considered together with fertility.
# Return:
#   (features, label), where features = [(day, symptom)...]
def genSingleCycleData():
    features = []
    sexProb = 0.15  # prob for sex symptom
    emotionProb = 0.3  # prob for emotion symptom
    epsilon = 0.05  # random noise scale
    probs = []  # probs for each sex
    fertility = genFertilities(cycleLength)
    for d in range(cycleLength):
        fertilityFactor = 1.
        if flipWithProb(emotionProb):  # emotion
            emotionIdx = np.random.randint(0, 3)
            emotionName = emotionImpacts.keys()[emotionIdx]
            fertilityFactor *= emotionImpacts[emotionName]
            features.append([d, emotionName])
        if flipWithProb(sexProb):  # have sex
            sexIdx = np.random.randint(0, 3)
            sexName = symptomNames[sexIdx]
            f = fertility[d] * fertilityFactor
            probs.append(f * sexRisks[sexName])
            features.append([d, sexName])
    finalProb = 1 - np.prod(1 - np.array(probs)) + np.random.randn() * epsilon
    finalProb = max(0, finalProb)
    # print("# finalProb: {}".format(finalProb))
    label = 1 if finalProb > 0.5 else 0
    return features, label


def genCycleData(num=0):
    data = [genSingleCycleData() for i in range(num)]
    return data

def convertToNdarray(data):
    numSymptom = len(symptomNames)
    dataNpy = np.zeros((len(data), 1 + numSymptom * cycleLength))
    for i, entry in enumerate(data):
        dataNpy[i][0] = entry[1]
        for symptom in entry[0]:
            day = symptom[0]
            symptomName = symptom[1]
            symptomIndex = symptomIndices[symptomName]
            dataNpy[i][1 + numSymptom * day + symptomIndex] = 1
    return dataNpy

def splitAndSave(dataNpy, trainPercent, devPercent, testPercent, dataDir):
    n = dataNpy.shape[0]

    np.random.shuffle(dataNpy)

    trainCount = int(n * trainPercent)
    devCount = int(n * devPercent)

    dataSplits = {
        "train": dataNpy[:trainCount, :],
        "dev": dataNpy[trainCount:trainCount+devCount, :],
        "test": dataNpy[trainCount+devCount:, :]
    }

    for k, v in dataSplits.items():
        splitDir = os.path.join(dataDir, k)
        os.mkdir(splitDir)
        splitFile = os.path.join(splitDir, k+".npy")
        np.save(splitFile, v)



if __name__ == "__main__":
    data = genCycleData(1)
    print("sample = {}".format(data))
    data = genCycleData(10)
    print("data = {}".format(data))
