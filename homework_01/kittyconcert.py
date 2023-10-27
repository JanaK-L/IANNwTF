import numpy as np

# - Aufgabe 2.1 -------------------------------------------------
from cat import Cat

cat1 = Cat("Kittosaurus Rex")
cat2 = Cat("Snowball IX")

cat1.intro()
cat1.greet(cat2.getName())

cat2.intro()
cat2.greet(cat1.getName())



# - Aufgabe 2.2 -------------------------------------------------
listeDerQuadrate = [x**2 for x in range(101)]
print(listeDerQuadrate)

listeDerGeradenQuadrate = [x**2 for x in range(101) if x % 2 == 0]
print(listeDerGeradenQuadrate)



# - Aufgabe 2.3 --------------------------------------------------
def create_MiauGenerator(anzahl:int):
    meinelist = (2**p for p in range(0, anzahl))
    for i in meinelist:
        yield "MIAU " * i   

miauGenerator = create_MiauGenerator(7)
for i in miauGenerator:
    print(i)



# - Aufgabe 2.4 --------------------------------------------------
# 2.4.1
sigma = 1
mü = 0
arr = np.random.normal(0, 1, (5, 5))
print(arr)


# 2.4.2
rows, cols = arr.shape
for i in range(cols):
    for j in range(rows):
        if arr[j,i] > 0.09:
            arr[j,i] = arr[j,i] ** 2
        else:
            arr[j,i] = 42    
print(arr)


# 2.4.3
print(arr[:, 3:4])