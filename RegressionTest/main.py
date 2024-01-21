import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

diabetes = load_diabetes()

x = diabetes.data
y = diabetes.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

reg = linear_model.LinearRegression()

reg.fit(x_train, y_train)

print("Regresyon Katsayısı : \n", reg.coef_)
print("Varyans Puanı : {}".format(reg.score(x_test, y_test)))

# Tahminleri oluşturun
y_pred = reg.predict(x_test)

print("Min yaş değeri:", np.min(x_train[:, 0]))
print("Max yaş değeri:", np.max(x_train[:, 0]))

# Scatter plot çiziminde yatay ekseni yaş, dikey ekseni hastalık seviyesi olarak belirtin
plt.scatter(x_test[:, 0], y_test, color='b', label='Gerçek Değerler')
plt.plot(x_test[:, 0], y_pred, color='k', label='Tahminler')

plt.xlabel('Yaş')  # x ekseni etiketi
plt.ylabel('Hastalık Seviyesi')  # y ekseni etiketi
plt.legend()  # etiketleri göster

plt.show()



