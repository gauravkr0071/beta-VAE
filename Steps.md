Same steps are used for IMS bearing dataset and CWRU ball bearing dataset

```py
STEP 1:-
Dependencies:-
1- Tensorflow 1.x/2.x
2- Keras
3- Cuda driver (optional)
4- Pandas
5- matplotlib
6- sklearn	
7- Numpy

code for installing dependencies-
in colab : pip install "library name"
and press RUN
for example : pip install keras
and press RUN

in CMD : pip install "library name"
and press enter
```
STEP 2: Importing libraries
```py

import numpy as np
import pandas as pd
%tensorflow_version 1.x
import tensorflow
import random
import keras.utils
import matplotlib
import matplotlib.pyplot as plt

```
```
STEP 3: path of dataset
// 
k=50000
bearing=np.loadtxt(r'/folder_in_which_your_py_is_placed/acd_bearing.dat', delimiter=',')
bearing=bearing.reshape(k,1)

piston=np.loadtxt(r'/folder_in_which_your_py_is_placed/acd_piston.dat', delimiter=',')
piston=piston.reshape(k,1)

nrv=np.loadtxt(r'/folder_in_which_your_py_is_placed/acd_nrv.dat', delimiter=',')
nrv=nrv.reshape(k,1)

liv=np.loadtxt(r'/folder_in_which_your_py_is_placed/acd_liv.dat', delimiter=',')
liv=liv.reshape(k,1)

lov=np.loadtxt(r'/folder_in_which_your_py_is_placed/acd_lov.dat', delimiter=',')
lov=lov.reshape(k,1)

flywheel=np.loadtxt(r'/folder_in_which_your_py_is_placed/acd_flywheel.dat', delimiter=',')
flywheel=flywheel.reshape(k,1)

healthy=np.loadtxt(r'/folder_in_which_your_py_is_placed/acd_healthy.dat', delimiter=',')
healthy=healthy.reshape(k,1)

riderbelt=np.loadtxt(r'/content/gdrive/MyDrive/dataacd/acd_riderbelt.dat', delimiter=',')
riderbelt=riderbelt.reshape(k,1)
```
```py
STEP 4: plotting any signal use below code , one category is shown for example

plt.rcParams["figure.figsize"] = (15,3)
plt.plot(flywheel[5000:8000,0])
plt.show()

```
```py
STEP 5: Processsing with min max algorithm

pc_pro=1500
pc_pro_test=150
cat=8
sam_pro=1500*cat
sam_pro_test=1500*cat
tk= 5000 #tp/10

k=10
bearing_pro=np.zeros((tk,1,2),dtype=np.float)#Normal_pro=bearing_pro

for i in range(0,5000,1):
    a=bearing[i*10:(i+1)*10,0] #normal_sliced=bearing
    a=a.reshape(k,1)
    max_num = np.where(a == np.amax(a))
    min_num= np.where(a == np.amin(a))
    min_i=min_num[0][0]
    max_i=max_num[0][0]
    if (min_i > max_i):
        bearing_pro[i,0,1]=a[min_i,0]
        bearing_pro[i,0,0]=a[max_i,0]
    else :
        bearing_pro[i,0,1]=a[max_i,0]
        bearing_pro[i,0,0]=a[min_i,0]
        
piston_pro=np.zeros((tk,1,2),dtype=np.float)        
for i in range(0,5000,1):
    a=piston[i*10:(i+1)*10,0]
    a=a.reshape(k,1)
    max_num = np.where(a == np.amax(a))
    min_num= np.where(a == np.amin(a))
    min_i=min_num[0][0]
    max_i=max_num[0][0]
    if (min_i > max_i):
        piston_pro[i,0,1]=a[min_i,0]
        piston_pro[i,0,0]=a[max_i,0]
    else :
        piston_pro[i,0,1]=a[max_i,0]
        piston_pro[i,0,0]=a[min_i,0]        
        
nrv_pro=np.zeros((tk,1,2),dtype=np.float)
for i in range(0,5000,1):
    a=nrv[i*10:(i+1)*10,0]
    a=a.reshape(k,1)
    max_num = np.where(a == np.amax(a))
    min_num= np.where(a == np.amin(a))
    min_i=min_num[0][0]
    max_i=max_num[0][0]
    if (min_i > max_i):
        nrv_pro[i,0,1]=a[min_i,0]
        nrv_pro[i,0,0]=a[max_i,0]
    else :
        nrv_pro[i,0,1]=a[max_i,0]
        nrv_pro[i,0,0]=a[min_i,0]  
        
liv_pro=np.zeros((tk,1,2),dtype=np.float)    
for i in range(0,5000,1):
    a=liv[i*10:(i+1)*10,0]
    a=a.reshape(k,1)
    max_num = np.where(a == np.amax(a))
    min_num= np.where(a == np.amin(a))
    min_i=min_num[0][0]
    max_i=max_num[0][0]
    if (min_i > max_i):
        liv_pro[i,0,1]=a[min_i,0]
        liv_pro[i,0,0]=a[max_i,0]
    else :
        liv_pro[i,0,1]=a[max_i,0]
        liv_pro[i,0,0]=a[min_i,0]  
        
lov_pro=np.zeros((tk,1,2),dtype=np.float)    
for i in range(0,5000,1):
    a=lov[i*10:(i+1)*10,0]
    a=a.reshape(k,1)
    max_num = np.where(a == np.amax(a))
    min_num= np.where(a == np.amin(a))
    min_i=min_num[0][0]
    max_i=max_num[0][0]
    if (min_i > max_i):
        lov_pro[i,0,1]=a[min_i,0]
        lov_pro[i,0,0]=a[max_i,0]
    else :
        lov_pro[i,0,1]=a[max_i,0]
        lov_pro[i,0,0]=a[min_i,0]  

flywheel_pro=np.zeros((tk,1,2),dtype=np.float)    
for i in range(0,5000,1):
    a=flywheel[i*10:(i+1)*10,0]
    a=a.reshape(k,1)
    max_num = np.where(a == np.amax(a))
    min_num= np.where(a == np.amin(a))
    min_i=min_num[0][0]
    max_i=max_num[0][0]
    if (min_i > max_i):
        flywheel_pro[i,0,1]=a[min_i,0]
        flywheel_pro[i,0,0]=a[max_i,0]
    else :
        flywheel_pro[i,0,1]=a[max_i,0]
        flywheel_pro[i,0,0]=a[min_i,0] 
        
        
healthy_pro=np.zeros((tk,1,2),dtype=np.float)    
for i in range(0,5000,1):
    a=healthy[i*10:(i+1)*10,0]
    a=a.reshape(k,1)
    max_num = np.where(a == np.amax(a))
    min_num= np.where(a == np.amin(a))
    min_i=min_num[0][0]
    max_i=max_num[0][0]
    if (min_i > max_i):
        healthy_pro[i,0,1]=a[min_i,0]
        healthy_pro[i,0,0]=a[max_i,0]
    else :
        healthy_pro[i,0,1]=a[max_i,0]
        healthy_pro[i,0,0]=a[min_i,0]             

        
        
riderbelt_pro=np.zeros((tk,1,2),dtype=np.float)    
for i in range(0,5000,1):
    a=riderbelt[i*10:(i+1)*10,0]
    a=a.reshape(k,1)
    max_num = np.where(a == np.amax(a))
    min_num= np.where(a == np.amin(a))
    min_i=min_num[0][0]
    max_i=max_num[0][0]
    if (min_i > max_i):
        riderbelt_pro[i,0,1]=a[min_i,0]
        riderbelt_pro[i,0,0]=a[max_i,0]
    else :
        riderbelt_pro[i,0,1]=a[max_i,0]
        riderbelt_pro[i,0,0]=a[min_i,0]             
```

```py
Step 6: Reshaping vector

bearing_pro=bearing_pro.reshape(tk,2)
bearing_pro=bearing_pro.reshape(tk*2,1)

nrv_pro=nrv_pro.reshape(tk,2)
nrv_pro=nrv_pro.reshape(tk*2,1)

piston_pro=piston_pro.reshape(tk,2)
piston_pro=piston_pro.reshape(tk*2,1)

lov_pro=lov_pro.reshape(tk,2)
lov_pro=lov_pro.reshape(tk*2,1)

liv_pro=liv_pro.reshape(tk,2)
liv_pro=liv_pro.reshape(tk*2,1)

flywheel_pro=flywheel_pro.reshape(tk,2)
flywheel_pro=flywheel_pro.reshape(tk*2,1)

healthy_pro=healthy_pro.reshape(tk,2)
healthy_pro=healthy_pro.reshape(tk*2,1)

riderbelt_pro=riderbelt_pro.reshape(tk,2)
riderbelt_pro=riderbelt_pro.reshape(tk*2,1)

```
```py
Step 7: converting 1D vector into image(2d) for traing and test data

random.seed(42)
for i in range(0,pc,1):
    len=bearing_pro.shape[0]-prod
    end=random.randint(0,len)
    a=bearing_pro[end:end+prod,0]    
    train[i,0,:,:]=a.reshape(ch,ch)
for i in range(pc,2*pc,1):
    len=piston_pro.shape[0]-prod
    end=random.randint(0,len)
    a=piston_pro[end:end+prod,0]
    train[i,0,:,:]=a.reshape(ch,ch)
for i in range(2*pc,3*pc,1):
    len=nrv_pro.shape[0]-prod
    end=random.randint(0,len)
    a=nrv_pro[end:end+prod,0]
    train[i,0,:,:]=a.reshape(ch,ch)
for i in range(3*pc,4*pc,1):
    len=liv_pro.shape[0]-prod
    end=random.randint(0,len)
    a=liv_pro[end:end+prod,0]
    train[i,0,:,:]=a.reshape(ch,ch)
for i in range(4*pc,5*pc,1):
    len=lov_pro.shape[0]-prod
    end=random.randint(0,len)
    a=lov_pro[end:end+prod,0]    
    train[i,0,:,:]=a.reshape(ch,ch)
for i in range(5*pc,6*pc,1):
    len=flywheel_pro.shape[0]-prod
    end=random.randint(0,len)
    a=flywheel_pro[end:end+prod,0]
    train[i,0,:,:]=a.reshape(ch,ch)
for i in range(6*pc,7*pc,1):
    len=healthy_pro.shape[0]-prod
    end=random.randint(0,len)
    a=healthy_pro[end:end+prod,0]
    train[i,0,:,:]=a.reshape(ch,ch)
for i in range(7*pc,8*pc,1):
    len=riderbelt_pro.shape[0]-prod
    end=random.randint(0,len)
    a=riderbelt_pro[end:end+prod,0]
    train[i,0,:,:]=a.reshape(ch,ch)    
    
for i in range(0,pc_test,1):
    len=bearing_pro.shape[0]-prod
    end=random.randint(0,len)
    a=bearing_pro[end:end+prod,0]
    test[i,0,:,:]=a.reshape(ch,ch)
for i in range(pc_test,2*pc_test,1):
    len=piston_pro.shape[0]-prod
    end=random.randint(0,len)
    a=piston_pro[end:end+prod,0]
    test[i,0,:,:]=a.reshape(ch,ch)
for i in range(2*pc_test,3*pc_test,1):
    len=nrv_pro.shape[0]-prod
    end=random.randint(0,len)
    a=nrv_pro[end:end+prod,0]
    test[i,0,:,:]=a.reshape(ch,ch)
for i in range(3*pc_test,4*pc_test,1):
    len=liv_pro.shape[0]-prod
    end=random.randint(0,len)
    a=liv_pro[end:end+prod,0]
    test[i,0,:,:]=a.reshape(ch,ch)
for i in range(4*pc_test,5*pc_test,1):
    len=lov_pro.shape[0]-prod
    end=random.randint(0,len)
    a=lov_pro[end:end+prod,0]
    test[i,0,:,:]=a.reshape(ch,ch)
for i in range(5*pc_test,6*pc_test,1):
    len=flywheel_pro.shape[0]-prod
    end=random.randint(0,len)
    a=flywheel_pro[end:end+prod,0]
    test[i,0,:,:]=a.reshape(ch,ch)
for i in range(6*pc_test,7*pc_test,1):
    len=healthy_pro.shape[0]-prod
    end=random.randint(0,len)
    a=healthy_pro[end:end+prod,0]
    test[i,0,:,:]=a.reshape(ch,ch)
for i in range(7*pc_test,8*pc_test,1):
    len=riderbelt_pro.shape[0]-prod
    end=random.randint(0,len)
    a=riderbelt_pro[end:end+prod,0]
    test[i,0,:,:]=a.reshape(ch,ch)    
 ```          
 
```
Step 8: output labels
y_train=np.zeros((sam,1),dtype=np.float)
y_test=np.zeros((sam_test,1),dtype=np.float)

y_train[0:pc,0]=0
y_train[pc:2*pc,0]=1
y_train[2*pc:3*pc,0]=2
y_train[3*pc:4*pc,0]=3
y_train[4*pc:5*pc,0]=4
y_train[5*pc:6*pc,0]=5
y_train[6*pc:7*pc,0]=6
y_train[7*pc:8*pc,0]=7



y_test[0:pc_test,0]=0
y_test[pc_test:2*pc_test,0]=1
y_test[2*pc_test:3*pc_test,0]=2
y_test[3*pc_test:4*pc_test,0]=3
y_test[4*pc_test:5*pc_test,0]=4
y_test[5*pc_test:6*pc_test,0]=5
y_test[6*pc_test:7*pc_test,0]=6
y_test[7*pc_test:8*pc_test,0]=7
print(y_train.shape)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train,8)
y_test = to_categorical(y_test, 8)
print(test.shape,train.shape)
print(y_train.shape,y_test.shape)

train=np.moveaxis(train,1,3)
test=np.moveaxis(test,1,3)
print(train.shape)

```


```py
Step 9: for plotting 2d image
'import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (4,4)
plt.imshow(train[1,:,:,:].reshape(50,50) )

plt.show()
```

```py
Step 10: importing some libraries


import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline

from scipy.stats import norm

import keras
from keras import layers
from keras.models import Model
from keras import metrics
#import tensorflow.compat.v1.keras.backend as K
#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
from keras import backend as K  

```
```py
step 11: fiest part of training VAE (encoder+ decoder)
img_shape = (ch, ch, 1)    # for MNIST
batch_size = 1
latent_dim =20  # Number of latent dimension parameters

# Encoder architecture: Input -> Conv2D*4 -> Flatten -> Dense
input_img = keras.Input(shape=img_shape)#(None, 28, 28, 1)
x=layers.BatchNormalization() (input_img)
x = layers.Conv2D(32, 3,
                  padding='same', 
                  activation='relu')(input_img) #(None, 28, 28, 32)

x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu',
                  strides=(2, 2))(x)#(None, 14, 14, 64)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)#(None, 14, 14, 64)                  

x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)#(None, 14, 14, 64)



# need to know the shape of the network here for the decoder
shape_before_flattening = K.int_shape(x)#shape_before_flattening (None, 14, 14, 64)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x) #(None, 32)


# Two outputs, latent mean and (log)variance
z_mu = layers.Dense(latent_dim)(x)#Tensor("dense_5/BiasAdd:0", shape=(None, 2), dtype=float32)
z_log_sigma = layers.Dense(latent_dim)(x)#Tensor("dense_5/BiasAdd:0", shape=(None, 2), dtype=float32)
# sampling function
def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),
                              mean=0., stddev=1)
    return z_mu + K.exp(z_log_sigma) * epsilon

# sample vector from the latent distribution
z = layers.Lambda(sampling,output_shape=(20,) )([z_mu, z_log_sigma]) #(None, 2)

# decoder takes the latent distribution sample as input
decoder_input = layers.Input(K.int_shape(z)[1:]) #K.int_shape(z)[1:]=(2,)  #K.int_shape(z)[0:]=(None, 2)
#decoder _input shape=(None, 2)
# Expand to 784 total pixels
#print(shape_before_flattening[1:]) =(14, 14, 64)
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input) #(None, 12544)


# reshape
x = layers.Reshape(shape_before_flattening[1:])(x) #(None, 14, 14, 64)

# use Conv2DTranspose to reverse the conv layers from the encoder
x = layers.Conv2DTranspose(32, 3,
                           padding='same', 
                           activation='relu',
                           strides=(2, 2))(x) # (None, 14, 14, 64)

x = layers.Conv2D(1, 3,
                  padding='same', 
                  activation='sigmoid')(x) #(None, 28, 28, 1)  




decoder = Model(decoder_input, x)
decoder.summary()

# apply the decoder to the sample from the latent distribution


z_decoded = decoder(z)#(None, 28, 28, 1)
# construct a custom layer to calculate the loss
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # Reconstruction loss
        
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        # KL divergence
        print(z_log_sigma.shape)
        kl_loss = -0.5e-5* K.sum(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # adds the custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# apply the custom loss to the input images and the decoded latent distribution sample
y = CustomVariationalLayer()([input_img, z_decoded])




vae = Model(input_img, y)

vae.compile(optimizer='rmsprop', loss=None)#must be greater than  -.0.0664
vae.fit(x=train, y=None,
        shuffle=True,
        epochs=300,
        batch_size=10,
        validation_data=(test, None))

```

- output
```cpp
WARNING:tensorflow:From /tensorflow-1.15.2/python3.7/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 20)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 40000)             840000    
_________________________________________________________________
reshape_1 (Reshape)          (None, 25, 25, 64)        0         
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 50, 50, 32)        18464     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 50, 50, 1)         289       
=================================================================
Total params: 858,753
Trainable params: 858,753
Non-trainable params: 0
_________________________________________________________________
(?, 20)
WARNING:tensorflow:From /tensorflow-1.15.2/python3.7/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
/tensorflow-1.15.2/python3.7/keras/engine/training_utils.py:819: UserWarning: Output custom_variational_layer_1 missing from loss dictionary. We assume this was done on purpose. The fit and evaluate APIs will not be expecting any data to be passed to custom_variational_layer_1.
  'be expecting any data to be passed to {0}.'.format(name))
WARNING:tensorflow:From /tensorflow-1.15.2/python3.7/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 12000 samples, validate on 1200 samples
Epoch 1/300
12000/12000 [==============================] - 239s 20ms/step - loss: 4306.4740 - val_loss: -0.0601
Epoch 2/300
12000/12000 [==============================] - 239s 20ms/step - loss: -0.0675 - val_loss: -0.0746
Epoch 3/300
12000/12000 [==============================] - 240s 20ms/step - loss: -0.0875 - val_loss: -0.0965
Epoch 4/300
12000/12000 [==============================] - 241s 20ms/step - loss: -0.1033 - val_loss: -0.1053
Epoch 5/300
12000/12000 [==============================] - 238s 20ms/step - loss: -0.1103 - val_loss: -0.1076
Epoch 6/300
12000/12000 [==============================] - 236s 20ms/step - loss: -0.1151 - val_loss: -0.1086
Epoch 7/300
12000/12000 [==============================] - 236s 20ms/step - loss: -0.1198 - val_loss: -0.1120
Epoch 8/300
12000/12000 [==============================] - 236s 20ms/step - loss: -0.1240 - val_loss: -0.1127
Epoch 9/300
12000/12000 [==============================] - 237s 20ms/step - loss: -0.1276 - val_loss: -0.1126
Epoch 10/300
12000/12000 [==============================] - 239s 20ms/step - loss: -0.1311 - val_loss: -0.1141
Epoch 11/300
12000/12000 [==============================] - 238s 20ms/step - loss: -0.1343 - val_loss: -0.1156
Epoch 12/300
12000/12000 [==============================] - 238s 20ms/step - loss: -0.1373 - val_loss: -0.1154
Epoch 13/300
 2390/12000 [====>.........................] - ETA: 3:06 - loss: -0.1433
```
```py
Step 12: 

for layer in vae.layers[:]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in vae.layers:
    print(layer, layer.trainable)

output-

<keras.engine.input_layer.InputLayer object at 0x7f0ecca27890> False
<keras.layers.convolutional.Conv2D object at 0x7f0ecca27b50> False
<keras.layers.convolutional.Conv2D object at 0x7f0ecca27fd0> False
<keras.layers.convolutional.Conv2D object at 0x7f0ec5a8e250> False
<keras.layers.convolutional.Conv2D object at 0x7f0ec5a4a850> False
<keras.layers.core.Flatten object at 0x7f0ec5a588d0> False
<keras.layers.core.Dense object at 0x7f0ec5a64fd0> False
<keras.layers.core.Dense object at 0x7f0ec59faa10> False
<keras.layers.core.Dense object at 0x7f0ec5a6ee10> False
<keras.layers.core.Lambda object at 0x7f0ec5a14bd0> False
<keras.engine.training.Model object at 0x7f0ec59c1d10> False
<__main__.CustomVariationalLayer object at 0x7f0ec59d9090> False

```

```py
Step 13:

lay1_1= layers.Dense(64, activation='relu')(z)


lay61   =layers.Dense(32, activation='relu')(lay1_1)
lay61   =layers.Dense(16, activation='relu')(lay61)
lay71=layers.Dense(8,activation='softmax')(lay61)

vae_f1=Model(input_img,lay71) 
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
vae_f1.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
vae_f1.summary()

output of summary

Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 50, 50, 1)    0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 50, 50, 32)   320         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 25, 25, 64)   18496       conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 25, 25, 64)   36928       conv2d_2[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 25, 25, 64)   36928       conv2d_3[0][0]                   
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 40000)        0           conv2d_4[0][0]                   
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 32)           1280032     flatten_1[0][0]                  
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 20)           660         dense_1[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 20)           660         dense_1[0][0]                    
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 20)           0           dense_2[0][0]                    
                                                                 dense_3[0][0]                    
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 64)           1344        lambda_1[0][0]                   
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 32)           2080        dense_5[0][0]                    
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 16)           528         dense_6[0][0]                    
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 8)            136         dense_7[0][0]                    
==================================================================================================
Total params: 1,378,112
Trainable params: 4,088
Non-trainable params: 1,374,024

```
```py
Step 15 : checking the alyers state

for layer in vae_f1.layers:
    print(layer, layer.trainable)
    
output
<keras.engine.input_layer.InputLayer object at 0x7f0ecca27890> False
<keras.layers.convolutional.Conv2D object at 0x7f0ecca27b50> False
<keras.layers.convolutional.Conv2D object at 0x7f0ecca27fd0> False
<keras.layers.convolutional.Conv2D object at 0x7f0ec5a8e250> False
<keras.layers.convolutional.Conv2D object at 0x7f0ec5a4a850> False
<keras.layers.core.Flatten object at 0x7f0ec5a588d0> False
<keras.layers.core.Dense object at 0x7f0ec5a64fd0> False
<keras.layers.core.Dense object at 0x7f0ec59faa10> False
<keras.layers.core.Dense object at 0x7f0ec5a6ee10> False
<keras.layers.core.Lambda object at 0x7f0ec5a14bd0> False
<keras.layers.core.Dense object at 0x7f0ebe2a9990> True
<keras.layers.core.Dense object at 0x7f0ebe2a97d0> True
<keras.layers.core.Dense object at 0x7f0ebe2ca190> True
<keras.layers.core.Dense object at 0x7f0ed259f110> True

```
```py
Step 16: final part of training that is calssification
vae_f1.fit(x=train, y=y_train,
        shuffle=True,
        epochs=100,
        batch_size=10,
        validation_data=(test,y_test)) 
        


```
