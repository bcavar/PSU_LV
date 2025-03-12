"""PRVI
import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,3,1]
y= [1,2,2,1,1]
plt.plot(x, y, linewidth=2, marker=".", markersize=10)
plt.axis([0,4,0,4])
plt.xlabel('X os')
plt.ylabel('Y os')
plt.title('Zad 1')
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("C:/Users/student/Desktop/mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)

mpg=data[:,0]
hp=data[:,3]
wt=data[:6]
cyl=data[:,1]



plt.scatter(mpg,hp)
plt.xlabel('MPG')
plt.ylabel('hp')
plt.title('ovisnost potrošnje automobila (mpg) o konjskim snagama (hp)')
plt.show()

min=100.0
max=0.0
br=0
suma=0.0

for mpg in data[:,0]:
    if max<mpg:
        max=mpg
    if min>mpg:
        min=mpg
    
    br +=1
    suma+=mpg

sv=suma/br
print("Min:",min,"Max:",max,"Srednja vrijednost:",sv)

minCyl=100.0
maxCyl=0.0
brCyl=1
sumaCyl=0.0

for cyl in data [:,1]:
    if cyl == 6:
        for mpg in data[:,0]:
         if maxCyl<mpg:
              maxCyl=mpg
         if minCyl>mpg:
             minCyl=mpg
    
         brCyl +=1
         sumaCyl+=mpg

svCyl=sumaCyl/brCyl
print("MinCyl:",minCyl,"MaxCyl:",maxCyl,"Srednja vrijednostCyl:",svCyl)
