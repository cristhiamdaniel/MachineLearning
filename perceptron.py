import numpy as np

class Percentron(object):
    """
    Parametros
    ----------------
    eta : float
        Tasa de aprendizaje (0.0 - 1.0)
    n_inter : int
        Pasos para entrenar los datos

    Atributos
    ----------------
    w_ : arreglo 1d
        Pesos despues del ajuste
    errors_ : lista
        Número de clasificaciones erróneas en cada época.
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        """ Ajuste de la data de entrenamiento

        Parametros
        ---------------
        X : {matriz} shape =[n_samples, n_features]
            Vectores de entrenamiento, donde
            n_samples es el numero de muestras y
            n_features es el numero dee caracteristicas

        y : array-like, shape = [n_samples]
            Valores objetivo

        Retornos
        ------------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self,X):
        """ Calculo de la red de entrada"""
        return np.dot(X,self.w_[1:]) + self.w_[0]
    def predict(self,X):
        """Retorna la etiqueta de la Clase despues del paso unitario"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)