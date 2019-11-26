
### A project implementing different methods of clustering

We implement two different methods of clustering using different methods of measuring distance


### Visualizing clusters

Here we visulize the different different clusters implemented in the Visual Studio solution. We start with the 2 dimensional data in clusters2.csv


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

Cluster2 = pd.read_csv("cluster2.csv")
Cluster2 = Cluster2.drop(Cluster2.columns[0], axis=1,)

def plotboy(Cluster,title):
    y = []

    for x in Cluster.values:
        x = [float(i) for i in x[0].split(',')]
        y.append(x)
    
    fig1, ax1 = plt.subplots()
    colors=["red", "blue", "green", "magenta", "yellow","purple"]
    for x in range(len(y)):
        p1 = []
        for i in range(len(y[x])-2):
            p1.append(Cluster2.values[int(y[x][i])][:])
        p1 = np.array(p1)
        
        ax1.scatter(*zip(*p1), c=colors[x])
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.set_title(title)
    plt.show()

Cluster = pd.read_fwf("Gcluster2-Euclid.csv", header=None, delimiter='|')
plotboy(Cluster,"Greedycluster using Euclidian distance on 2d data")


```


![png](output_1_0.png)



```python
Cluster = pd.read_fwf("Gcluster2-Manhatt.csv", header=None, delimiter='|')
plotboy(Cluster,"Greedycluster using Manhattan distance on 2d data")

```


![png](output_2_0.png)



```python
Cluster = pd.read_fwf("Gcluster2-Mahal.csv", header=None, delimiter='|')
plotboy(Cluster,"Greedycluster using Mahalanobis distance on 2d data")
```


![png](output_3_0.png)



```python
Cluster = pd.read_fwf("Ccluster2-Mahal.csv", header=None, delimiter='|')
plotboy(Cluster,"Convergencecluster using Mahalanobis distance on 2d data")
```


![png](output_4_0.png)



```python
Cluster = pd.read_fwf("Ccluster2-Euclid.csv", header=None, delimiter='|')
plotboy(Cluster,"Convergencecluster using Euclidean distance on 2d data")
```


![png](output_5_0.png)



```python
Cluster = pd.read_fwf("Ccluster2-Manhatt.csv", header=None, delimiter='|')
plotboy(Cluster,"Convergencecluster using Manhattan distance on 2d data")
```


![png](output_6_0.png)


Now we visualize 4 dimensional data from clusters4.csv, we use a three dimensionall graph and we have size of the points represent the fourth dimension.


```python
from mpl_toolkits.mplot3d import Axes3D

Cluster4 = pd.read_csv("cluster4.csv")
Cluster4 = Cluster4.drop(Cluster2.columns[0], axis=1,)

def plotboy4d(Cluster,title):
    y = []

    for x in Cluster.values:
        x = [float(i) for i in x[0].split(',')]
        y.append(x)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    colors=["red", "blue", "green", "black", "yellow","purple"]
    for x in range(len(y)):
        x1,x2,x3,x4 = [],[],[],[]

        for i in range(len(y[x])-4):
            x1.append(Cluster4.values[int(y[x][i])][0])
            x2.append(Cluster4.values[int(y[x][i])][1])
            x3.append(Cluster4.values[int(y[x][i])][2])
            x4.append(Cluster4.values[int(y[x][i])][3])
   
        for i,j,k, size in zip(x1,x2,x3,x4):
            ax1.scatter(i,j,k, c=colors[x], s = np.log(size**100), edgecolors='none', alpha=0.7)
    
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.set_zlabel(r'$x_3$')
    ax1.set_title(title)
    plt.show()

Cluster = pd.read_fwf("Gcluster4-Euclid.csv", header=None, delimiter='|')
plotboy4d(Cluster,"Greedycluster using Euclidian distance on 4d data")

```


![png](output_8_0.png)



```python
Cluster = pd.read_fwf("Gcluster4-Manhatt.csv", header=None, delimiter='|')
plotboy4d(Cluster,"Greedycluster using Manhattan distance on 4d data")
```


![png](output_9_0.png)



```python
Cluster = pd.read_fwf("Gcluster4-Mahal.csv", header=None, delimiter='|')
plotboy4d(Cluster,"Greedycluster using Mahalanobis distance on 4d data")
```


![png](output_10_0.png)



```python
Cluster = pd.read_fwf("Ccluster4-Mahal.csv", header=None, delimiter='|')
plotboy4d(Cluster,"Convergence using Mahalanobis distance on 4d data")
```


![png](output_11_0.png)



```python
Cluster = pd.read_fwf("Ccluster4-Euclid.csv", header=None, delimiter='|')
plotboy4d(Cluster,"Convergence using Euclidean distance on 4d data")
```


![png](output_12_0.png)



```python
Cluster = pd.read_fwf("Ccluster4-Manhatt.csv", header=None, delimiter='|')
plotboy4d(Cluster,"Convergence using Manhattan distance on 4d data")
```


![png](output_13_0.png)

