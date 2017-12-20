#Dense NET1
#Autoencoder
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Flatten, Reshape, Dropout, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from keras import backend as K
import tensorflow as tf

def tuple_generator(generator):
    for batch in generator:
        yield (batch, batch)

np.random.seed(777)
if (len(sys.argv)<3):
    print("Input arguments:")
    print("1. Train images path")
    print("2. Test images path")
    exit()
PreTrainImagesPath=sys.argv[1]
PreTestImagesPath=sys.argv[2]
TrainImagesPath=sys.argv[1]
TestImagesPath=sys.argv[2]
img_width, img_height = 128, 128
epochs = 35
batch_size = 32
dropout_rate = 0.3
latent_dim = 101

datagen=ImageDataGenerator(rescale=1./255)
pretrain_generator = datagen.flow_from_directory(
        PreTrainImagesPath,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None)
train_samples = pretrain_generator.n

validation_generator = datagen.flow_from_directory(
        PreTestImagesPath,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None)
nb_validation_samples = validation_generator.n

input_img = Input(shape=(img_width, img_height, 3))
x = Flatten(input_shape=(img_width, img_height, 3))(input_img)
encoded = (Dense(units=300, activation='sigmoid', name='dens_sigmoid_1'))(x)
x = Dense(img_width*img_height*3, activation='sigmoid')(encoded)
decoded = Reshape((img_width, img_height, 3), input_shape=(img_width*img_height*3,))(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')
print (autoencoder.summary())
autoencoder.fit_generator(
        tuple_generator(pretrain_generator),
        steps_per_epoch=train_samples/batch_size,
        epochs=epochs
        )

plot_model(autoencoder, to_file='DenseAutoEncoder.png', show_shapes=True, show_layer_names=False, rankdir='LR')

autoencoder.save('DenseAutoEncoderNET1')
autoencoder.save_weights('DenseAutoEncoderNET1_weights')

#Without pretrained weights
datagen=ImageDataGenerator(samplewise_center=True,
    samplewise_std_normalization=True)

train_generator = datagen.flow_from_directory(
        TrainImagesPath,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical', seed=777)
train_count = train_generator.n
test_generator = datagen.flow_from_directory(
        TestImagesPath,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',shuffle=False)
test_count = test_generator.n
#%%
model = Sequential()
model.add(Flatten(input_shape=(128,128,3),name='flatten'))
model.add(Dense(units=300, activation='sigmoid', name='dens_sigmoid_1'))
model.add(Dense(units=101, activation='softmax'))
#model.load_weights('DenseAutoEncoderNET1_weights', by_name=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#%%
early_stopping=EarlyStopping(monitor='acc', patience=3, verbose=0, mode='auto')
t0=time.time()
model.fit_generator(train_generator,
        steps_per_epoch=train_count/batch_size,
        epochs=epochs,
        callbacks=[early_stopping])
t1=time.time()
loss_and_metrics = model.evaluate_generator(test_generator, steps=test_count/batch_size)
print('Accuracy =',loss_and_metrics[1])
print('Time =',(t1-t0))
#%%
#With pretrained weights
model = Sequential()
model.add(Flatten(input_shape=(128,128,3),name='flatten'))
model.add(Dense(units=300, activation='sigmoid', name='dens_sigmoid_1'))
model.add(Dense(units=101, activation='softmax'))
model.load_weights('DenseAutoEncoderNET1_weights', by_name=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#%%
early_stopping=EarlyStopping(monitor='acc', patience=3, verbose=0, mode='auto')
t0=time.time()
model.fit_generator(train_generator,
        steps_per_epoch=train_count/batch_size,
        epochs=epochs,
        callbacks=[early_stopping])
t1=time.time()
loss_and_metrics = model.evaluate_generator(test_generator, steps=test_count/batch_size)
print('Accuracy =',loss_and_metrics[1])
print('Time =',(t1-t0))
#%%
K.clear_session()
sess = tf.Session()
K.set_session(sess)

#Dense NET2
#Autoencoder
# first layer's autoencoder
input_img = Input(shape=(img_width, img_height, 3))
x = Flatten(input_shape=(img_width, img_height, 3))(input_img)
encoded = (Dense(units=1000, activation='sigmoid', name='dens_sigmoid_1'))(x)
x = Dense(img_width*img_height*3, activation='sigmoid')(encoded)
decoded = Reshape((img_width, img_height, 3), input_shape=(img_width*img_height*3,))(x)

autoencoder1 = Model(input_img, decoded)

autoencoder1.compile(optimizer='adadelta', loss='mse')
print (autoencoder1.summary())
autoencoder1.fit_generator(
        tuple_generator(pretrain_generator),
        steps_per_epoch=train_samples/batch_size,
        epochs=epochs
        )
plot_model(autoencoder1, to_file='DenseAutoEncoder2_l1.png', show_shapes=True, show_layer_names=True, rankdir='LR')
autoencoder1.save('DenseAutoEncoderNET2_l1')
autoencoder1.save_weights('DenseAutoEncoderNET2_weights_l1')

encoder1 = Sequential()
encoder1.add(Flatten(input_shape=(img_width, img_height, 3)))
encoder1.add(Dense(units=1000, activation='sigmoid', name='dens_sigmoid_1'))
encoder1.load_weights('DenseAutoEncoderNET2_weights_l1', by_name=True)
encoder1.compile(optimizer='adadelta', loss='mse')
encoder1_out = (encoder1.predict_generator(tuple_generator(pretrain_generator),steps=train_samples/batch_size))
print (encoder1_out.shape)

# second layer's autoencoder
x = Input(shape=(1000,))
encoded = (Dense(units=500, activation='tanh', name='dens_tanh_1'))(x)
decoded = Dense(1000, activation='tanh')(encoded)

autoencoder2 = Model(x, decoded)
autoencoder2.compile(optimizer='adadelta', loss='mse')
print (autoencoder2.summary())
autoencoder2.fit(
        encoder1_out, encoder1_out,
        batch_size=batch_size,
        epochs=epochs
        )
plot_model(autoencoder2, to_file='DenseAutoEncoder2_l2.png', show_shapes=True, show_layer_names=True, rankdir='LR')
autoencoder2.save('DenseAutoEncoderNET2_l2')
autoencoder2.save_weights('DenseAutoEncoderNET2_weights_l2')

encoder2 = Sequential()
encoder2.add(Dense(units=500, activation='tanh', input_shape=(1000,), name='dens_tanh_1'))
encoder2.load_weights('DenseAutoEncoderNET2_weights_l2', by_name=True)
encoder2.compile(optimizer='adadelta', loss='mse')
encoder2_out = encoder2.predict(encoder1_out)

# third layer's autoencoder
x = Input(shape=(500,))
encoded = (Dense(units=300, activation='relu', name='dens_relu_1'))(x)
decoded = Dense(500, activation='relu')(encoded)

autoencoder3 = Model(x, decoded)
autoencoder3.compile(optimizer='adadelta', loss='mse')
print (autoencoder3.summary())
autoencoder3.fit(
        encoder2_out, encoder2_out,
        batch_size=batch_size,
        epochs=epochs
        )
plot_model(autoencoder3, to_file='DenseAutoEncoder2_l3png', show_shapes=True, show_layer_names=True, rankdir='LR')
autoencoder3.save('DenseAutoEncoderNET2_l3')
autoencoder3.save_weights('DenseAutoEncoderNET2_weights_l3')
#Without pretrained weights
model = Sequential()
model.add(Flatten(input_shape=(128,128,3)))
model.add(Dense(units=1000, activation='sigmoid', name='dens_sigmoid_1'))
model.add(Dense(units=500, activation='tanh', name='dens_tanh_1'))
model.add(Dense(units=300, activation='relu', name='dens_relu_1'))
model.add(Dense(units=101, activation='softmax'))
#model.load_weights('DenseAutoEncoderNET2_weights_l1', by_name=True)
#model.load_weights('DenseAutoEncoderNET2_weights_l2', by_name=True)
#model.load_weights('DenseAutoEncoderNET2_weights_l3', by_name=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#%%
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='acc', patience=3, verbose=0, mode='auto')
import time
t0=time.time()
model.fit_generator(train_generator,
        steps_per_epoch=train_count/batch_size,
        epochs=epochs,
        callbacks=[early_stopping])
t1=time.time()
loss_and_metrics = model.evaluate_generator(test_generator, steps=test_count/batch_size)
print('Accuracy =',loss_and_metrics[1])
print('Time =',(t1-t0))
#%%
#With pretrained weights
model = Sequential()
model.add(Flatten(input_shape=(128,128,3)))
model.add(Dense(units=1000, activation='sigmoid', name='dens_sigmoid_1'))
model.add(Dense(units=500, activation='tanh', name='dens_tanh_1'))
model.add(Dense(units=300, activation='relu', name='dens_relu_1'))
model.add(Dense(units=101, activation='softmax'))
model.load_weights('DenseAutoEncoderNET2_weights_l1', by_name=True)
model.load_weights('DenseAutoEncoderNET2_weights_l2', by_name=True)
model.load_weights('DenseAutoEncoderNET2_weights_l3', by_name=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#%%
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='acc', patience=3, verbose=0, mode='auto')
import time
t0=time.time()
model.fit_generator(train_generator,
        steps_per_epoch=train_count/batch_size,
        epochs=epochs,
        callbacks=[early_stopping])
t1=time.time()
loss_and_metrics = model.evaluate_generator(test_generator, steps=test_count/batch_size)
print('Accuracy =',loss_and_metrics[1])
print('Time =',(t1-t0))
#%%
K.clear_session()
sess = tf.Session()
K.set_session(sess)
#Conv NET1
#Autoencoder
# first layer's autoencoder
input_img = Input(shape=(img_width, img_height, 3))
x = Conv2D(32, (3, 3), activation='tanh', padding='same', input_shape=(img_width,img_height,3), name='conv_tanh_1')(input_img)
x = Conv2D(32, (3, 3), activation='tanh', padding='same', name='conv_tanh_2')(x)
x = MaxPooling2D((2, 2), padding='same', name='maxpool_1')(x)
encoded = Dropout((0.25), name='dropout_1')(x)
x = UpSampling2D((2, 2))(encoded)
x = Conv2D(32, (3, 3), activation='tanh', padding='same')(x)
decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)

autoencoder1 = Model(input_img, decoded)

autoencoder1.compile(optimizer='adadelta', loss='mse')
print (autoencoder1.summary())
autoencoder1.fit_generator(
        tuple_generator(pretrain_generator),
        steps_per_epoch=train_samples/batch_size,
        epochs=epochs
        )
plot_model(autoencoder1, to_file='ConvAutoEncoder2_l1.png', show_shapes=True, show_layer_names=True, rankdir='LR')
autoencoder1.save('ConvAutoEncoderNET1_l1')
autoencoder1.save_weights('ConvAutoEncoderNET1_weights_l1')

encoder1 = Sequential()
encoder1.add(Conv2D(32, (3, 3), activation='tanh', padding='same', input_shape=(img_width,img_height,3), name='conv_tanh_1'))
encoder1.add(Conv2D(32, (3, 3), activation='tanh', padding='same', name='conv_tanh_2'))
encoder1.add(MaxPooling2D((2, 2), padding='same', name='maxpool_1'))
encoder1.add(Dropout((0.25), name='dropout_1'))
encoder1.add(Flatten(input_shape=(64, 64, 32)))
encoder1.load_weights('ConvAutoEncoderNET1_weights_l1', by_name=True)
encoder1.compile(optimizer='adadelta', loss='mse')
encoder1_out = (encoder1.predict_generator(tuple_generator(pretrain_generator),steps=train_samples/batch_size))
print (encoder1_out.shape)

# second layer's autoencoder
x = Input(shape=(131072,))
x = Dense(units=512, activation='tanh', name='dense_tanh_1')(x)
encoded = Dropout((0.5), name='dropout_2')(x)
x = Dense(units=512, activation='tanh')(encoded)
decoded = Dense(units=131072)(x)

autoencoder2 = Model(x, decoded)
autoencoder2.compile(optimizer='adadelta', loss='mse')
print (autoencoder2.summary())
autoencoder2.fit(
        encoder1_out, encoder1_out,
        batch_size=batch_size,
        epochs=epochs
        )
plot_model(autoencoder2, to_file='ConvAutoEncoder2_l2.png', show_shapes=True, show_layer_names=True, rankdir='LR')
autoencoder2.save('ConvAutoEncoderNET1_l2')
autoencoder2.save_weights('ConvAutoEncoderNET1_weights_l2')
#Without pretrained weights
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='tanh', padding='same', input_shape=(128,128,3), name='conv_tanh_1'))
model.add(Conv2D(32, (3, 3), activation='tanh', padding='same', name='conv_tanh_2'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same', name='maxpool_1'))
model.add(Dropout(0.25, name='dropout_1'))
model.add(Flatten(name='flat_1'))
model.add(Dense(512, activation='tanh', name='dense_tanh_1'))
model.add(Dropout(0.5, name='dropout_2'))
model.add(Dense(101, activation='softmax', name='dense_softmax_1'))
#model.load_weights('ConvAutoEncoderNET1_weights_l1', by_name=True)
#model.load_weights('ConvAutoEncoderNET1_weights_l2', by_name=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#%%
early_stopping=EarlyStopping(monitor='acc', patience=3, verbose=0, mode='auto')
t0=time.time()
model.fit_generator(train_generator,
        steps_per_epoch=train_count/batch,
        epochs=epochs,
        callbacks=[early_stopping])
t1=time.time()
loss_and_metrics = model.evaluate_generator(test_generator, steps=test_count/batch)
print('Accuracy =',loss_and_metrics[1])
print('Time =',(t1-t0))
#%%
#With pretrained weights
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='tanh', padding='same', input_shape=(128,128,3), name='conv_tanh_1'))
model.add(Conv2D(32, (3, 3), activation='tanh', padding='same', name='conv_tanh_2'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same', name='maxpool_1'))
model.add(Dropout(0.25, name='dropout_1'))
model.add(Flatten(name='flat_1'))
model.add(Dense(512, activation='tanh', name='dense_tanh_1'))
model.add(Dropout(0.5, name='dropout_2'))
model.add(Dense(101, activation='softmax', name='dense_softmax_1'))
model.load_weights('ConvAutoEncoderNET1_weights_l1', by_name=True)
model.load_weights('ConvAutoEncoderNET1_weights_l2', by_name=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#%%
early_stopping=EarlyStopping(monitor='acc', patience=3, verbose=0, mode='auto')
t0=time.time()
model.fit_generator(train_generator,
        steps_per_epoch=train_count/batch,
        epochs=epochs,
        callbacks=[early_stopping])
t1=time.time()
loss_and_metrics = model.evaluate_generator(test_generator, steps=test_count/batch)
print('Accuracy =',loss_and_metrics[1])
print('Time =',(t1-t0))
#%%