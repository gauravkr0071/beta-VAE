### __CWRU__

- __Importing Necessary Library__
```py
%tensorflow_version 1.x
import numpy as np
import pandas as pd
import random
import keras.utils
import matplotlib
import matplotlib.pyplot as plt

```

__Importing Dataset: CWRU__
```py
Normal=pd.read_csv('/content/gdrive/MyDrive/Myfaultdata/Normal.csv',header=None)
Normal.plot()
plt.show()

Normal=Normal.values
Ball=pd.read_csv('/content/gdrive/MyDrive/Myfaultdata/Ball.csv',header=None)
Ball.plot()
plt.show()
Ball=Ball.values
Inner=pd.read_csv('/content/gdrive/MyDrive/Myfaultdata/Inner.csv',header=None)
Inner.plot()
plt.show()
Inner=Inner.values
Outer=pd.read_csv('/content/gdrive/MyDrive/Myfaultdata/Outer.csv',header=None)
Outer.plot()
plt.show()
Outer=Outer.values
print(Normal.shape,Ball.shape,Inner.shape,Outer.shape)
```
