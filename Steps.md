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

```

