import matplotlib.pyplot as plt
import numpy as np
import soundfile
import librosa, librosa.display
import os
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.metrics._classification import confusion_matrix


def train_graph_curve(model, parameter,  X_train, y_train, noise=False, pitch=False, shift=False):
    print("[^-] Calculating learning curve... ")
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax1.set_ylabel("Score")
    ax1.set_xlabel("Number of samples")
    if noise or pitch or shift:
        ax1.set_title("Learning " + parameter + " rate with sound tunes")
    else:
        ax1.set_title("Learning " + parameter + " rate")

    print("[^-] Plotting learning curve...")
    # Plot learning curve
    ax1.grid()
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    ax1.legend(loc="best")

    #  loss
    print("[^-] Plotting loss graph...")
    ax2.grid()
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Iterations")
    ax2.set_title("Loss " + parameter + " graph\n")
    ax2.plot(model.loss_curve_)

    plt.show()


def train_graph_matrix(H, parameter, y_true, y_pred, noise=False, pitch=False, shift=False):

    print("[^-] Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    display_labels = H.classes_

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    disp1 = disp.plot(include_values=True, xticks_rotation='horizontal', cmap=plt.cm.magma)
    if noise or pitch or shift:
        disp1.ax_.set_title("Confusion " + parameter + " matrix rate with sound tunes")
    else:
        disp1.ax_.set_title("Confusion " + parameter + " matrix")
    plt.show()


def graph_spectr_wave(filename1, filename2):
    X = soundfile.SoundFile(filename2).read(dtype="float32")  # выводим график спектрограммы
    stft = np.abs(librosa.stft(X))
    fig = plt.Figure()
    ax = fig.add_subplot(1, 1, 1)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), ax=ax,
                                 y_axis='log', x_axis='time')

    y, sr = librosa.load(filename1)  # выводим график вейвформы
    fig2 = plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(y, sr=sr)

    if not os.path.isdir("graphics"):
        os.mkdir("graphics")

    fig.savefig("graphics/chroma.png")  # сохраняем графики
    fig2.savefig("graphics/wave.png")


def validation_graph(model, X_train, y_train, valid_param, valid_range=[]):
    train_scores, test_scores = validation_curve(
        model, X_train, y_train, param_name=valid_param, param_range=valid_range,
        scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel(valid_param)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(valid_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(valid_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(valid_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(valid_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
