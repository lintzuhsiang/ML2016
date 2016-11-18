#coding = utf-8
from keras.layers import Input, Dense,Activation, Convolution2D, MaxPooling2D, UpSampling2D,Flatten,Merge,Dropout,BatchNormalization
from keras.models import Model,load_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
import numpy as np
import pickle
import os
import sys
import six.moves.cPickle as pickle
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.backend import set_image_dim_ordering
set_image_dim_ordering('th')
os.environ["THEANO_FLAGS"]="device=gpu0"

input_img = Input(shape=(3,32, 32))

x = Convolution2D(256, 3, 3, activation='sigmoid', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(128, 3, 3, activation='sigmoid', border_mode='same')(x)
#x = MaxPooling2D((2, 2), border_mode='same')(x)
#x = Convolution2D(50, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)


x = Convolution2D(128, 3, 3, activation='sigmoid', border_mode='same')(encoded)
#x = UpSampling2D((2, 2))(x)
#x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(256 ,3, 3, activation='sigmoid', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics = ['accuracy'])
autoencoder.summary()

x_train = pickle.load(open(sys.argv[1]+'all_label.p','rb'))
x_train = np.asarray(x_train).reshape(5000,3,32,32)
x_train = x_train.astype('float32') / 255.
x_unlabel = pickle.load(open(sys.argv[1]+'all_unlabel.p','rb'))
x_unlabel = np.asarray(x_unlabel).reshape(45000,3,32,32)
x_unlabel = x_unlabel.astype('float32') / 255.

y_train = np.zeros(500)
y_label = []
for i in range(0,10):
    y_train[:] = i;
    y_label = np.append(y_label,y_train)
y_train = np_utils.to_categorical(y_label,10)


noise_factor = 0.5
#x_train = x_train + noise_factor*np.random.normal(loc = 0.0,scale = 1.0)
#x_train = np.clip(x_train, 0., 1.)
check = ModelCheckpoint(filepath =sys.argv[1]+'middle1.h5',monitor = 'val_acc',verbose = 0,save_best_only=True,save_weights_only=True,mode='max')
autoencoder.fit(x_train, x_train,nb_epoch=80,batch_size=200,shuffle=True, validation_split = 0.3 ,callbacks=[check])
result = autoencoder.evaluate(x_train,x_train,batch_size=100)
print("Test accuracy: ",result[1])



#save model
autoencoder.save(sys.argv[1]+'middle1.h5')
print("autoencoder saved")


encoder_model = Sequential()
encoder_model.add(Convolution2D(64,3,3,input_shape=(3,32,32)))
encoder_model.add(MaxPooling2D((2,2)))
encoder_model.add(Dropout(0.25))

encoder_model.add(Convolution2D(64,3,3,border_mode='same'))
encoder_model.add(MaxPooling2D((2,2)))
encoder_model.add(Dropout(0.25))

encoder_model.add(Convolution2D(32,3,3,border_mode='same'))
encoder_model.add(MaxPooling2D((2,2)))
encoder_model.add(Dropout(0.25))

encoder_model.add(Flatten())
encoder_model.add(Dense(512))
encoder_model.add(GaussianNoise( 0.5 ))
encoder_model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.99, weights=None))
encoder_model.add(Activation('sigmoid'))
encoder_model.add(Dropout(0.5))
encoder_model.add(Dense(512))
encoder_model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.99, weights=None))
encoder_model.add(Activation('sigmoid'))
encoder_model.add(Dense(10))
encoder_model.add(Activation('softmax'))

encoder_model.summary()
encoder_model.compile(loss='binary_crossentropy',optimizer='Adadelta',metrics=['accuracy'])

encoder_model.fit(x_train,y_train,batch_size=100,nb_epoch=80,validation_split =0.2)
newresult = encoder_model.evaluate(x_train,y_train,batch_size=100,verbose=0)
print(newresult)

#save model
encoder_model.save(sys.argv[1]+'middle2.h5')
print("middle2.h5 saved")	


k=0
for k in range(0,30):
	autoencoder = load_model(sys.argv[1]+'middle2.h5')
	thredhold = 0.95
	x_predict = autoencoder.predict(x_unlabel,verbose = 0)
	x_predict_maxvalue = np.amax(x_predict, axis=1)
	index = np.array([],dtype=int)
	print(x_predict_maxvalue.shape)
	for i in range(0,len(x_predict)):
		if (x_predict_maxvalue[i] > thredhold):
			index = np.append(index,np.array(i))

	x_predict_argmax = np.argmax(x_predict[index],axis = 1).astype('int')
	y_label= np.concatenate((y_label, x_predict_argmax), axis=0)
	
	x_train = np.concatenate((x_train,x_unlabel[index].astype('int')),axis = 0)
	x_train = x_train.reshape(-1,3,32,32)
	y_train = np_utils.to_categorical(y_label,10)

	check = ModelCheckpoint(filepath =sys.argv[1] + sys.argv[2],monitor = 'val_acc',verbose = 0,save_best_only=True,save_weights_only=True,mode='max')

	encoder_model.fit(x_train,y_train,batch_size=100,nb_epoch=80,validation_split =0.2 , callbacks=[check])
	result = encoder_model.evaluate(x_train,y_train,batch_size=100,verbose=0)
	print("Test accuracy: ",result)
	
	#save model
	encoder_model.save(sys.argv[1] + sys.argv[2])
	print("output_model saved")

	k = k+1
	print("finish",k,"iteration")


del encoder_model








