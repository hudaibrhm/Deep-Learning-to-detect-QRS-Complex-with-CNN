from __future__ import division
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras.models import Sequential

# Import
import numpy as np
import pandas as pd
np.random.seed(1337) # untuk menstabilkan akurasi training

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv('datafix.csv')

# dimensi
n = data.shape[0]
p = data.shape[1]

# buat data dengan np.array
data = data.values


# Training and test data
train_start = 0
train_end = int(np.floor(0.7*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# subset data X dan y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
x_test = data_test[:, 1:]
y_test = data_test[:, 0]
# definisi panjang qrs
max_qrs = 25
max_qrs1 = 30
# membangun model
embedding_length = 50
model = Sequential()
model.add(Embedding(max_qrs1, embedding_length, input_length=max_qrs))
model.add(Conv1D(batch_size=200, filters=50, kernel_size=4, padding='same', activation='sigmoid'))
model.add(AveragePooling1D(batch_size=200,pool_size=2))
model.add(Conv1D(batch_size=200, filters=50, kernel_size=4, padding='same', activation='sigmoid'))
model.add(Flatten())
model.add(Dense(1,batch_size=200, activation='relu',kernel_initializer="random_uniform",bias_initializer='zeros'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1)
outputs = [layer.output for layer in model.layers]	
# evaluasi pakai X_test dan y_test serta hitung akurasi

test = model.predict(x_test)
sukses = 0
sukses1 = 0
for i in range(0,360):
	if(test[i]>0.5):
		sukses=sukses+1
	
for j in range (361,722):
	if(test[j]<0.5):
		sukses1=sukses1+1
hasil=sukses1+sukses
jumlah=723
akurasi=(hasil/jumlah)*100
print("Akurasi: %.2f%%" %(akurasi))
