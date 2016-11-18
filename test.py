#coding = utf-8
from keras import layers
from keras.models import Sequential , load_model
from keras.layers import Dense, Activation,Merge
import pickle
import os
os.environ["THEANO_FLAGS"]="device=gpu0"
import numpy as np
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten,UpSampling2D
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import sys
from keras.backend import set_image_dim_ordering
set_image_dim_ordering('th')

loaded_model = load_model(sys.argv[1]+sys.argv[2])

test = pickle.load(open(sys.argv[1]+'test.p','rb'))

X_test = np.asarray(test['ID']).reshape(1,10000)


tmp=np.zeros((10000,3073))
for i in range(10000):
	tmp[i] = np.append(test['ID'][i],test['data'][i])
tmp = tmp[:,1:] 
tmp = np.asarray(tmp).reshape(10000,3,32,32)

loaded_model.compile(loss='categorical_crossentropy',optimizer='Adadelta',metrics=['accuracy'])

predict2 = loaded_model.predict_classes(tmp,batch_size=100,verbose=0)
print(predict2.shape)
print('loaded')

output = np.array([[0,predict2[0]]])
for i in range(1,10000):
	output=np.append(output,[[i,predict2[i]]],axis = 0)
np.savetxt(sys.argv[3],output,delimiter=',',fmt='%i',comments='',header="ID,class")
print("csv saved.")