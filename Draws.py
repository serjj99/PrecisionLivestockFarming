import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.tanh(x)



x_ = np.linspace(0, 5, 24)

Error_train = [5,4,3.1,2.3,1.5,1,0.9,0.8,0.6,0.5,0.42,0.35,0.32,0.3,0.27,0.25,0.235,0.23,0.22,0.219,0.218,0.217,0.216,0.215]
Error_test = [5,4.2,3.3,2.5,1.7,1.2,1.1,1,0.8,0.65,0.6,0.58,0.56,0.6,0.65,0.68,0.7,0.77,0.86,0.98,1.1,1.3,1.45,1.65]

plt.figure(figsize=(6,6))
# plt.title('Función logística')
plt.plot(x_,Error_train, color='blue', label = 'Error entrenamiento')
plt.plot(x_,Error_test, color='red', label = 'Error test')
plt.scatter(x_,Error_train, color='blue', s=10)
plt.scatter(x_,Error_test, color='red', s=10)
plt.plot([x_[12],x_[12]],[0,8], '--', color='black')
plt.xlim((0,5))
plt.ylim((0,5))
plt.legend()
plt.ylabel('Error')
plt.xlabel('Número de entrenamientos')
plt.savefig('Images/overfitting.png')
plt.show()
