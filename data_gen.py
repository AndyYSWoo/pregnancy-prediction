#!/usr/bin/env python3
"""Synthetic 24-day cycle generator for the graph-imputation prototype."""
import os
import numpy as np

cycleLength = 24
sexRisks = {"unprotected_sex": 0.5, "protected_sex": 0.2, "withdrawl_sex": 0.3}
emotionImpacts = {"emotion_happy": 1.2, "emotion_neutral": 1.0, "emotion_sad": 0.8}
symptomNames = list(sexRisks) + list(emotionImpacts)
symptomIndices = {name: idx for idx, name in enumerate(symptomNames)}
defaultPatientStructureStrength = 0.8


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def _softmax(values):
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def genPatientTraits(rng=None):
    """Draw stable latent event-propensity, category preference, and risk traits."""
    rng = np.random.default_rng() if rng is None else rng
    return {
        "sex_activity": float(rng.normal()),
        "emotion_reporting": float(rng.normal()),
        "sex_preference": rng.normal(size=3),
        "emotion_preference": rng.normal(size=3),
        "fertility_susceptibility": float(rng.normal()),
    }


def genFertilities(length=cycleLength):
    fertility = np.array([
        1, 1, 1, 1, 1, 1, 2, 3, 10, 12, 13, 11, 10, 8, 10, 9, 5,
        3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 0, 0, 0,
    ], dtype=float)
    if not 1 <= length <= fertility.size:
        raise ValueError("cycle length must be between 1 and 34 days")
    return fertility[:length] / fertility.max() * 0.8


def genSingleCycleData(
    rng=None, length=cycleLength, patient_traits=None,
    structure_strength=defaultPatientStructureStrength,
):
    if not 0 <= structure_strength <= 2:
        raise ValueError("patient structure strength must be in [0, 2]")
    rng = np.random.default_rng() if rng is None else rng
    patient_traits = genPatientTraits(rng) if patient_traits is None else patient_traits
    events = np.zeros((length, len(symptomNames)), dtype=np.float32)
    probabilities = []
    fertility = genFertilities(length)
    sex_probability = float(_sigmoid(
        np.log(0.15 / 0.85) + structure_strength * patient_traits["sex_activity"]
    ))
    emotion_probability = float(_sigmoid(
        np.log(0.3 / 0.7) + structure_strength * patient_traits["emotion_reporting"]
    ))
    sex_category_probabilities = _softmax(
        structure_strength * np.asarray(patient_traits["sex_preference"])
    )
    emotion_category_probabilities = _softmax(
        structure_strength * np.asarray(patient_traits["emotion_preference"])
    )
    susceptibility = float(np.exp(np.clip(
        structure_strength * 0.35 * patient_traits["fertility_susceptibility"],
        -1.5, 1.5,
    )))
    for day in range(length):
        fertility_factor = 1.0
        if rng.random() < emotion_probability:
            emotion = int(rng.choice(3, p=emotion_category_probabilities))
            events[day, 3 + emotion] = 1
            fertility_factor *= list(emotionImpacts.values())[emotion]
        if rng.random() < sex_probability:
            sex = int(rng.choice(3, p=sex_category_probabilities))
            events[day, sex] = 1
            probabilities.append(
                min(
                    0.95,
                    fertility[day] * fertility_factor *
                    list(sexRisks.values())[sex] * susceptibility,
                )
            )
    probability = 1 - np.prod(1 - np.asarray(probabilities))
    probability = max(0.0, probability + rng.normal(0, 0.05))
    return events, int(probability > 0.5)


def genCycleData(
    num=0, seed=None, length=cycleLength,
    structure_strength=defaultPatientStructureStrength, return_traits=False,
):
    if not 0 <= structure_strength <= 2:
        raise ValueError("patient structure strength must be in [0, 2]")
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(num):
        traits = genPatientTraits(rng)
        events, label = genSingleCycleData(
            rng, length, traits, structure_strength=structure_strength
        )
        data.append((events, label, traits) if return_traits else (events, label))
    return data


def convertToNdarray(data):
    result = np.zeros((len(data), 1 + cycleLength * len(symptomNames)), dtype=np.float32)
    for i, (events, label) in enumerate(data):
        if np.asarray(events).shape != (cycleLength, len(symptomNames)):
            raise ValueError("event array must have shape (24, 6)")
        result[i, 0] = label
        result[i, 1:] = np.asarray(events).reshape(-1)
    return result


def splitAndSave(dataNpy, trainPercent, devPercent, testPercent, dataDir):
    if not np.isclose(trainPercent + devPercent + testPercent, 1):
        raise ValueError("split percentages must sum to 1")
    os.makedirs(dataDir, exist_ok=True)
    data = np.asarray(dataNpy).copy()
    np.random.default_rng().shuffle(data)
    train_count = int(len(data) * trainPercent)
    dev_count = int(len(data) * devPercent)
    splits = {
        "train": data[:train_count],
        "dev": data[train_count:train_count + dev_count],
        "test": data[train_count + dev_count:],
    }
    for name, split in splits.items():
        split_dir = os.path.join(dataDir, name)
        os.makedirs(split_dir, exist_ok=True)
        np.save(os.path.join(split_dir, name + ".npy"), split)
    return splits


if __name__ == "__main__":
    print(convertToNdarray(genCycleData(10, seed=0)).shape)
