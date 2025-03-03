"""PRVI
def total_euro(rs, euro):
    return print("total_euro:", rs*euro)

print("Unesi sate:")
rs=int(input())
print("Unesi euro:")
euro=int(input())

ukupno = rs * euro

print("Ukupno:" , ukupno)

total_euro(rs,euro)
"""
""" DRUGI
broj=float()
print("Unesi broj 0.0<broj<1.0:")
while (0.1>broj) or (1.0<broj):
    try:
        broj=float(input())
    except :
        print("Krivi unos")

if broj >= 0.9:
    print("A")
elif broj >= 0.8: 
    print("B")
elif broj >= 0.7:
    print("C")
elif broj >= 0.6:
    print("D")
elif broj < 0.6:
    print("F")
"""

""" TRECI
lista =[]
brojac=int()
sv=float()
zbroj=float()
i=0

while True:
    broj=input("Unesi broj:")
    if broj == "Done":
        break
    lista.append(broj)
    fbroj= float(broj)
    zbroj += fbroj
    brojac +=1

sv = zbroj / brojac
print(lista)
print("Ukupnan broj:",brojac)
print("Srednja vrijednost:",sv)
print("Min:",min(lista))
print("Max:",max(lista))
lista.sort()
print("Sortirano:",lista)
""" 

ime=(input("Unesite ime datoteke:"))
line="X-DSPAM-Confidence:"
f=open(ime)

for line in f:
    line=line.rsplit()
    print(line)

f.close()