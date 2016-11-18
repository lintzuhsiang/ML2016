#coding=UTF-8
from keras.models import Sequential , load_model
from keras.layers import Dense, Activation,Merge,Dropout,BatchNormalization, Convolution2D, MaxPooling2D, Flatten
import six.moves.cPickle as pickle
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.callbacks import ModelCheckpoint
import numpy as np 
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import sys
import os
os.environ["THEANO_FLAGS"]="device=gpu0"
from keras.backend import set_image_dim_ordering
set_image_dim_ordering('th')

x_train = pickle.load(open(sys.argv[1]+'all_label.p','rb'))
x_train = np.asarray(x_train)
x_train = x_train.astype('float32') 
x_train /= 255.

x_unlabel = pickle.load(open(sys.argv[1]+'all_unlabel.p','rb'))
x_unlabel = np.asarray(x_unlabel)
x_unlabel = x_unlabel.astype('float32').reshape(45000,3,32,32)
x_unlabel /= 255.


model2 = Sequential()
model2.add(Convolution2D(64,3,3,input_shape=(3,32,32)))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.25))
model2.add(Convolution2D(64,3,3,border_mode='same'))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.25))
model2.add(Convolution2D(64,3,3,border_mode='same'))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.25))
model2.add(Convolution2D(64,3,3,border_mode='same'))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(512))
model2.add(GaussianNoise( 0.5 ))
model2.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.99, weights=None))
model2.add(Activation('sigmoid'))
model2.add(Dropout(0.5))
model2.add(Dense(512))
model2.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.99, weights=None))
model2.add(Activation('sigmoid'))
model2.add(Dense(10))
model2.add(Activation('softmax'))

model2.summary()
model2.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


#create y_train
y_train = np.zeros(500)
y_label = []
for i in range(0,10):
    y_train[:] = i;
    y_label = np.append(y_label,y_train)
y_train = np_utils.to_categorical(y_label,10)

#shuffle data
data_shuffle = np.concatenate((x_train.reshape(5000,3072),y_train),axis = 1)
x_train = data_shuffle[:,:3072].reshape(5000,3,32,32)
y_train = data_shuffle[:,3072:]
np.random.shuffle(data_shuffle)

check = ModelCheckpoint(filepath = sys.argv[1]+sys.argv[2],monitor = 'val_acc',verbose = 0,save_best_only=True,save_weights_only=True,mode='max')

model2.fit(x_train,y_train,batch_size=100,nb_epoch=80,validation_split =0.2 , callbacks=[check])
result = model2.evaluate(x_train,y_train,batch_size=100,verbose=0)
print("Test accuracy: ",result[1])
#save model
model2.save(sys.argv[1]+sys.argv[2])

print("model saved")


#semi-supervised iteration
k=0
for k in range(0,20):
	loaded_model = load_model(sys.argv[1]+sys.argv[2])
	x_predict = loaded_model.predict(x_unlabel,verbose=0)
	thredhold = 0.95
	x_predict_maxvalue = np.amax(x_predict, axis=1)

	index = np.array([],dtype=int)
	for i in range(0,len(x_predict)):
		if (x_predict_maxvalue[i] > thredhold):
			index = np.append(index,np.array(i))


	x_predict_argmax = np.argmax(x_predict[index],axis = 1).astype('int')
	y_label= np.concatenate((y_label, x_predict_argmax), axis=0)
	
	x_train = np.concatenate((x_train,x_unlabel[index].astype('int')),axis = 0)
	x_train = x_train.reshape(-1,3,32,32)
	y_train = np_utils.to_categorical(y_label,10)

	print('x_train.shape:',x_train.shape)
	print('y_train.shape:',y_train.shape)

	loaded_model.compile(loss='categorical_crossentropy',optimizer='Adadelta',metrics=['accuracy'])
			
	check = ModelCheckpoint(filepath =sys.argv[1]+sys.argv[2],monitor = 'val_acc',verbose = 0,save_best_only=True,save_weights_only=True,mode='max')
	loaded_model.fit(x_train,y_train,batch_size=100,nb_epoch=80,validation_split =0.2)
	newresult = loaded_model.evaluate(x_train,y_train,batch_size=100,verbose=0)
	print(newresult)

	#save model
	loaded_model.save(sys.argv[1]+sys.argv[2])
	
	print("model saved")	
	k = k+1
	print("finish",k,"iteration")

