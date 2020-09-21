from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data
from sklearn.model_selection import GridSearchCV
from draw_graph import validation_graph

'''
Файл для экспериментов над нейросетью, на данный момент реализован только поиск по сетке
'''


def search(parameter, model_params, valid_param, noise=False,
           pitch=False, shift=False, t_size=0.25, valid=False, valid_range=[]):
    '''
    Функция поиска по сетке.
    Передаем параметр(имя модели),
    Словарь параметров по которым происходит поиск
    (каждый из которых обязательно записан в виде листа, либо другого промежутка),
    Размер тестовой выборки (часть от всей)
    '''
    X_train, X_test, y_train, y_test = load_data(parameter, test_size=t_size)  # загружаем данные

    print("[>] Number of training samples:", X_train.shape[0])
    print("[>] Number of testing samples:", X_test.shape[0])
    print("[>] Number of features:", X_train.shape[1])
    model = MLPClassifier()

    if valid:
        print("[^-] Plotting validation curve... ")
        validation_graph(model, X_train, y_train, valid_param, valid_range)

    else:
        clf = GridSearchCV(model, model_params, verbose=50, n_jobs=-1,
                           scoring='accuracy')  # ищем лучший набор параметров
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        print("[~] Accuracy: {:.2f}%".format(accuracy * 100) + "\n")

        print(clf.best_params_)  # выводим луший набор параметров и его точность
        print(clf.best_score_)


'''
M - Male
G - Gender
F - Female
'''

model_Mparams = {
    'activation': ['tanh', 'logistic', 'identity'],  # функция активации
    'alpha': [0.03],  # параметр регуляризации
    'batch_size': ['auto'],  # Размер мини-партий для стохастических оптимизаторов
    'epsilon': [1.25e-08],  # Значение для численной стабильности
    'hidden_layer_sizes': (400,),  # количество нейронов в скрытом слое
    'learning_rate': ['adaptive'],  # скорость обучения
    'max_iter': [500, ]  # максимальное количество итераций
}

model_Gparams = {
    'activation': ['tanh', 'logistic', 'identity'],
    'alpha': [0.09, 0.03, 0.01],
    'batch_size': [256, 'auto'],
    'epsilon': [1e-08],
    'hidden_layer_sizes': [(300,), (400,)],
    'learning_rate': ['adaptive'],
    'max_iter': [500]
}

model_Fparams = {
    'activation': ['tanh', 'logistic', 'identity'],
    'alpha': [0.09, 0.06, 0.01, 0.001],
    'batch_size': [256, 'auto'],
    'epsilon': [1.3e-08],
    'hidden_layer_sizes': [(300,), (400,)],
    'learning_rate': ['adaptive'],
    'max_iter': [500],
}

search("male", model_Mparams, noise=True, pitch=True, shift=True)


