from neuron.choice import sigmoid, sigmoid_prime, J, J_derivative
import numpy as np

def compute_grad_analytically(neuron, X, y, J_prime=J_derivative):
    """
    Analytical derivative of target function.
    neuron - object of class Neuron
    X - vertical input matrix (n, m).
    y - correct answers on X (m, 1)
    J_prime - function считающая производные целевой функции по ответам

    Возвращает вектор размера (m, 1)
    """

    z = neuron.summatory(X)
    y_hat = neuron.activation(z)

    # derivative chain
    dy_dyhat = J_prime(y, y_hat)  # 1st gear in chain
    dyhat_dz = neuron.activation_function_derivative(z)  # 2nd gear in chain
    dz_dw = X  # 3rd gear in chain

    grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw)
    grad = grad.T
    return grad


def compute_grad_numerically(neuron, X, y, J=J, eps=10e-6):
    """
    Численная производная целевой функции.
    neuron - объект класса Neuron с вертикальным вектором весов w,
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений,
    y - правильные ответы для тестовой выборки X,
    J - целевая функция, градиент которой мы хотим получить,
    eps - размер $\delta w$ (малого изменения весов).
    """

    w_0 = neuron.w
    num_grad = np.zeros(w_0.shape)

    for i in range(len(w_0)):
        old_wi = neuron.w[i].copy()
        # Меняем вес
        neuron.w[i] = old_wi + eps
        J_up = J(neuron, X, y)
        neuron.w[i] = old_wi - eps
        J_down = J(neuron, X, y)

        # Считаем новое значение целевой функции и вычисляем приближенное значение градиента
        num_grad[i] = (J_up - J_down) / (2 * eps)

        # Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать ошибки округления
        neuron.w[i] = old_wi

    # проверим, что не испортили нейрону веса своими манипуляциями
    assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА"
    return num_grad



