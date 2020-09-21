import soundfile
import numpy as np
import librosa, librosa.display
import glob
import os
from shutil import copy
from sklearn.model_selection import train_test_split

int2genders = {  # Словарь гендеров датасета для названий сэмплов
    "01": "male",
    "02": "female"
}

AVAILABLE_GENDERS = {  # Доступные гендеры для распознавания нейросети

    "male",
    "female"
}

int2emotion = {  # Словарь эмоций датасета для названий сэмплов
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

AVAILABLE_EMOTIONS = {  # Доступные эмоции для распознавания нейросети
    "angry",
    "sad",
    "neutral",
    "happy"
}


def pitch(data, sample_rate):
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype('float32'),
                                       sample_rate, n_steps=pitch_change,
                                       bins_per_octave=bins_per_octave)
    return data


def shift(data):

   s_range = int(np.random.uniform(low=-5, high=5)*500)
   data = np.roll(data, s_range)
   return data


def noise(data):
   """
   Добавление белого шума
   """
   noise_amp = 0.005*np.random.uniform()*np.amax(data)
   data = data.astype('float32') + noise_amp * np.random.normal(size=data.shape[0])
   return data


def extract_feature(file_name, **kwargs):
    '''
    Функция извлечения признаков MFCC, CHROMA, MEL (CONTRAST, TONNETZ)
    Принимает на вход сэмпл
    Возвращает набор признаков этого семпла
    '''
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    noise = kwargs.get("noise")
    shift = kwargs.get("shift")
    pitch = kwargs.get("pitch")
    with soundfile.SoundFile(file_name) as sound_file:
        # X, sample_rate = librosa.load(file_name, dtype="float32")
        X = sound_file.read(dtype="float32")  # Подготовка файла для работы пакета librosa
        sample_rate = sound_file.samplerate
        '''
        Если был упомянут признак - извлекаем его
        '''
        if noise:
            noise_amp = 0.005 * np.random.uniform() * np.amax(X)
            X = X.astype('float32') + noise_amp * np.random.normal(size=X.shape[0])
        if shift:
            s_range = int(np.random.uniform(low=-5, high=5) * 500)
            X = np.roll(X, s_range)
        if pitch:
            bins_per_octave = 12
            pitch_pm = 2
            pitch_change = pitch_pm * 2 * (np.random.uniform())
            X = librosa.effects.pitch_shift(X.astype('float32'),
                                            sample_rate, n_steps=pitch_change,
                                            bins_per_octave=bins_per_octave)
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))  # добавляем признак в массив
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result


def load_data(directory, parameter, test_size=0.2, noise=False, pitch=False, shift=False):
    '''
    Функция чтения датасета и составления
    списка признаков и соответствующих им эмоций/гендеров
    '''
    X, y = [], []
    if parameter == "gender":
        for file in glob.glob(directory):  # проходим по датасету
            basename = os.path.basename(file)
            gender = int2genders[basename.split("-")[0]]  # код гендера закодирован в первой (нулевой) части имени
            if gender not in AVAILABLE_GENDERS:
                continue
            features = extract_feature(file, mfcc=True, chroma=True, mel=True,
                                       contrast=True, tonnetz=True, noise=noise, pitch=pitch, shift=shift)
            #  извлекаем признаки
            X.append(features)  # записываем их
            y.append(gender)  # записываем соответсвующий гендер/эмоцию
        return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
    if parameter == "male":
        for file in glob.glob(directory):
            basename = os.path.basename(file)
            emotion = int2emotion[basename.split("-")[2]]  # код эмоции закодирован в третьей части имени
            id = basename.split("-")[0]
            if emotion not in AVAILABLE_EMOTIONS:
                continue
            if id != '01':
                continue
            features = extract_feature(file, mfcc=True, chroma=True, mel=True,
                                       contrast=True, tonnetz=True, noise=noise, pitch=pitch, shift=shift)
            X.append(features)
            y.append(emotion)
        return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
    if parameter == "female":
        for file in glob.glob(directory):
            basename = os.path.basename(file)
            emotion = int2emotion[basename.split("-")[2]]
            id = basename.split("-")[0]
            if emotion not in AVAILABLE_EMOTIONS:
                continue
            if id != '02':
                continue
            features = extract_feature(file, mfcc=True, chroma=True, mel=True,
                                       contrast=True, tonnetz=True, noise=noise, pitch=pitch, shift=shift)
            X.append(features)
            y.append(emotion)
        return train_test_split(np.array(X), y, test_size=test_size, random_state=7)


def load_dir(directory, parameter, noise=False, pitch=False, shift=False):
    '''
    Функция чтения датасета и составления
    списка признаков и соответствующих им эмоций/гендеров
    '''
    X, y = [], []
    indexes = []
    i = 0
    if parameter == "gender":
        for file in glob.glob(directory):  # проходим по датасету
            basename = os.path.basename(file)
            gender = int2genders[basename.split("-")[0]]  # код гендера закодирован в первой (нулевой) части имени
            if gender not in AVAILABLE_GENDERS:
                continue
            features = extract_feature(file, mfcc=True, chroma=True, mel=True,
                                       contrast=True, tonnetz=True, noise=noise, pitch=pitch, shift=shift)
            #  извлекаем признаки
            X.append(features)  # записываем их
            y.append(gender)  # записываем соответсвующий гендер/эмоцию
            indexes.append(i)
            i += 1
        return np.array(X), y, indexes
    if parameter == "male":
        for file in glob.glob(directory):
            basename = os.path.basename(file)
            emotion = int2emotion[basename.split("-")[2]]  # код эмоции закодирован в третьей части имени
            id = basename.split("-")[0]
            if emotion not in AVAILABLE_EMOTIONS:
                continue
            if id != '01':
                continue
            features = extract_feature(file, mfcc=True, chroma=True, mel=True,
                                       contrast=True, tonnetz=True, noise=noise, pitch=pitch, shift=shift)
            X.append(features)
            y.append(emotion)
            indexes.append(i)
            i += 1
        return np.array(X), y, indexes
    if parameter == "female":
        for file in glob.glob(directory):
            basename = os.path.basename(file)
            emotion = int2emotion[basename.split("-")[2]]
            id = basename.split("-")[0]
            if emotion not in AVAILABLE_EMOTIONS:
                continue
            if id != '02':
                continue
            features = extract_feature(file, mfcc=True, chroma=True, mel=True,
                                       contrast=True, tonnetz=True, noise=noise, pitch=pitch, shift=shift)
            X.append(features)
            y.append(emotion)
            indexes.append(i)
            i += 1
        return np.array(X), y, indexes


def search_error_samples(directory, parameter, error_indexes):
    i = 0
    if parameter == "gender":
        for file in glob.glob(directory):  # проходим по датасету
            basename = os.path.basename(file)
            gender = int2genders[basename.split("-")[0]]  # код гендера закодирован в первой (нулевой) части имени
            if gender not in AVAILABLE_GENDERS:
                continue
            if i in error_indexes:
                copy(file, "error_data")
            i += 1

    if parameter == "male":
        for file in glob.glob(directory):
            basename = os.path.basename(file)
            emotion = int2emotion[basename.split("-")[2]]  # код эмоции закодирован в третьей части имени
            id = basename.split("-")[0]
            if emotion not in AVAILABLE_EMOTIONS:
                continue
            if id != '01':
                continue
            if i in error_indexes:
                copy(file, "error_data")
            i += 1
    if parameter == "female":
        for file in glob.glob(directory):
            basename = os.path.basename(file)
            emotion = int2emotion[basename.split("-")[2]]
            id = basename.split("-")[0]
            if emotion not in AVAILABLE_EMOTIONS:
                continue
            if id != '02':
                continue
            if i in error_indexes:
                copy(file, "error_data")
            i += 1


def transform_samples(directory_in, directory_out, noise=False, pitch=False, shift=False):
    for file in glob.glob(directory_in):
        basename = os.path.basename(file)
        with soundfile.SoundFile(file) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            if noise:
                noise_amp = 0.005 * np.random.uniform() * np.amax(X)
                X = X.astype('float32') + noise_amp * np.random.normal(size=X.shape[0])
            if shift:
                s_range = int(np.random.uniform(low=-5, high=5) * 500)
                X = np.roll(X, s_range)
            if pitch:
                bins_per_octave = 12
                pitch_pm = 2
                pitch_change = pitch_pm * 2 * (np.random.uniform())
                X = librosa.effects.pitch_shift(X.astype('float32'),
                                                sample_rate, n_steps=pitch_change,
                                                bins_per_octave=bins_per_octave)
        soundfile.write(directory_out + basename, X, sample_rate)