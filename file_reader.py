from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

def getMnistData(verbose=False):
    # importar o conjunto de dados MNIST
    data, labels = fetch_openml('mnist_784', version=1, data_home = './resources', return_X_y=True, as_frame=False)
    #dataset = fetch_openml("MNIST Original", data_home='./resources')
    #(data, labels) = (dataset.data, dataset.target)

    if (verbose):
        # Exibir algumas informações do dataset MNIST
        print("[INFO] Número de imagens: {}".format(data.shape[0]))
        print("[INFO] Pixels por imagem: {}".format(data.shape[1]))

        # escolher um índice aleatório do dataset e exibir
        # a imagem e label correspondente
        np.random.seed(17)
        randomIndex = np.random.randint(0, data.shape[0])
        print("[INFO] Imagem aleatória do MNIST com label " + (labels[randomIndex]) + ':')
        plt.imshow(data[randomIndex].reshape((28,28)), cmap="Greys")
        plt.show()

    return data, labels 