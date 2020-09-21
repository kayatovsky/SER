from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data, load_dir, search_error_samples, transform_samples
from draw_graph import train_graph_curve, train_graph_matrix
from multiprocessing import Process

import os
import pickle


def save_model(model, parameter):
    print("[/] Saving model...")
    if not os.path.isdir("result"):  # создаем директорию для сохранения модели
        os.mkdir("result")

    if parameter == "gender":  # сохраняем модели и выводим графики тренировки
        pickle.dump(model, open("result/mlp_classifier_gender.model", "wb"))

    elif parameter == "male":
        pickle.dump(model, open("result/mlp_classifier_male.model", "wb"))

    elif parameter == "female":
        pickle.dump(model, open("result/mlp_classifier_female.model", "wb"))
    print("[/] Model saved\n")


def train(parameter, model_params, noise=False, pitch=False, shift=False,
          t_size=0.25, save_param=False, partial=False, dir_train="", dir_test=""):
    '''
    Функция тренировки модели
    Передается имя модели (параметр), словарь параметров, размер тестовой выборки
    '''
    print("[~] Downloading data...")
    if partial:
        X_train, y_train, _ = load_dir(dir_train, parameter)
        X_test, y_test, _ = load_dir(dir_test, parameter)
    else:
        X_train, X_test, y_train, y_test = load_data("totaldata/Actor_*/*.wav", parameter,
                                                     test_size=t_size, pitch=pitch, shift=shift, noise=noise)
    print("[~] Data downloaded\n")

    print("[>] Number of training samples:", X_train.shape[0])
    print("[>] Number of testing samples:", X_test.shape[0])
    print("[>] Number of features:", X_train.shape[1])

    # создаем модель (MLPClassifier (MultiLayerPerceptronClassifier) - считается лучшим классификатором для задач SER)
    model = MLPClassifier(**model_params)
    print("[.] Training the " + parameter + " model...")
    H = model.fit(X_train, y_train)  # тренируем модель

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)  # Вычисляем точность
    if noise or pitch or shift:
        print("[~] Accuracy " + parameter + " with sound tunes" + " : {:.2f}%".format(accuracy * 100) + "\n")
    else:
        print("[~] Accuracy " + parameter + " : {:.2f}%".format(accuracy * 100) + "\n")
    # train_graph_curve(model, parameter,  X_train, y_train, noise, pitch, shift)
    train_graph_matrix(model, parameter, y_test, y_pred, noise, pitch, shift)

    if save_param:
        save_model(model, parameter)


def test(parameter, dir_test):
    print("[~] Testing " + parameter + " model...")
    if parameter == "gender":
        model = pickle.load(open("result/mlp_classifier_gender.model", "rb"))

    elif parameter == "male":
        model = pickle.load(open("result/mlp_classifier_male.model", "rb"))

    elif parameter == "female":
        model = pickle.load(open("result/mlp_classifier_female.model", "rb"))

    print("[~] Downloading data...")
    X_test, y_test, indexes = load_dir(dir_test, parameter)
    print("[~] Data downloaded\n")
    print("[>] Number of testing samples:", X_test.shape[0])

    error_indexes = []
    print("[~] Predicting...")

    print("[~] Searching error samples...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)  # Вычисляем точность

    train_graph_matrix(model, parameter, y_test, y_pred)

    print("[~] Accuracy: {:.2f}%".format(accuracy * 100) + "\n")
    y_pred = []
    for i in indexes:
        predict = model.predict(X_test[i].reshape(1, -1))
        y_pred.append(predict)
        if y_pred[i] != y_test[i]:
            error_indexes.append(i)
    search_error_samples(dir_test, parameter, error_indexes)

    print("[!] Error indexes:\n")
    print(error_indexes)

'''
Параметры в словарях ниже - оптимальные для текущего датасета для каждой модели индивидуально,
найденные с помощью поиска по сетке (файл exp.py)
'''

modelG_params = {  # Что значат эти параметры описано в файле exp.py
    'alpha': 0.001,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (250,),
    'learning_rate': 'adaptive',
    'max_iter': 500
}

modelM_params = {
    'activation': 'tanh',
    'alpha': 0.03,
    'batch_size': 'auto',
    'epsilon': 1.25e-08,
    'hidden_layer_sizes': (400,),
    'learning_rate': 'adaptive',
    'max_iter': 500,
}

modelF_params = {
    'activation': 'tanh',
    'alpha': 0.03,
    'batch_size': 'auto',
    'epsilon': 1.25e-08,
    'hidden_layer_sizes': (400,),
    'learning_rate': 'adaptive',
    'max_iter': 500,
}


if __name__ == '__main__':
    # p1 = Process(target=train, args=("male", modelM_params, True, True, True))
    # p2 = Process(target=train, args=("male", modelM_params))
    # # p3 = Process(target=train, args=("female", modelF_params, True))
    # # p4 = Process(target=train, args=("female", modelF_params))
    #
    # p1.start()
    # p2.start()
    # # p3.start()
    # # p4.start()
    #
    # p1.join()
    # p2.join()
    # # p3.join()
    # # p4.join()

    transform_samples("totaldata/Actor_01/*.wav", "transformed_data/Actor_01/", shift=True)