""" PRVI
import pandas as pd
import numpy as np

mtcars = pd.read_csv('mtcars.csv')

top5_mpg=mtcars.sort_values(by='mpg',ascending=True).head(5)
print(top5_mpg[['car','mpg']])

low3_mpg=mtcars[mtcars['cyl'] == 8].sort_values(by='mpg',ascending=True).tail(3)
print(low3_mpg[['car','mpg','cyl']])

avg6=mtcars[mtcars['cyl']==6]['mpg'].mean()
print(f"Srednja vrijednost: {avg6:.2f} mpg")

avg4=mtcars[(mtcars['cyl']==4)&(mtcars['wt']<=2.2)&(mtcars['wt']>=2.0)]['mpg'].mean()
print(f"Srednja vrijednost izmedu 2000 i 2200lbs: {avg4:.2f} mpg")

manual=mtcars[mtcars['am'] == 1].shape[0]
automatic=mtcars[mtcars['am'] == 0].shape[0]
print(f"Broj manualnih: {manual}")
print(f"Broj automatic: {automatic}")

auto100ks=mtcars[(mtcars['am']==0)&(mtcars['hp']>=100)].shape[0]
print(f"Automatik preko 100 konja: {auto100ks}")

mtcars['wt_kg']=mtcars['wt']*453.592
print(f"Masa automobila u kg:\n {mtcars[['car','wt','wt_kg']]}")"""

""" DRUGI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mtcars = pd.read_csv("mtcars.csv")

plt.figure(figsize=(10, 6))
sns.barplot(x='cyl', y='mpg', data=mtcars, ci=None, palette='viridis')
sns.boxplot(x='cyl', y='wt', data=mtcars)
plt.title('Prosjecna potrosnja automobila s 4, 6 i 8 cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Prosječna potrošnja (mpg)')
plt.show()"""
