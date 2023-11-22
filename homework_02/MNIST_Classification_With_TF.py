import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

# - Aufgabe 2.1 Laden des Datensets -------------------------------------
(test, train), info = tfds.load("mnist", split=["train", "test"], as_supervised = True, with_info = True)  # Test und Training Samples als Tuples (features, label) laden

#print("\n\nInformationen zum Datenset:\n", info)

'''
• How many training/test images are there?
  Test = 10.000 Samples
  Training = 60.000 Samples
• What’s the image shape?
  Die Bilder haben eine Shape von (28, 28, 1), wobei die Bilder 28x28 Pixel beinhalten
  und die 1 dafür steht, dass die Bilder nur über einen Kanal (Grauwerte) verfügen. Panchromatisch vermutlich?
• What range are pixel values in?
  Der dtype der Images ist uint8 (8 Bit unsigned Integer). Demnach ist 0 der kleinste und 255 der höchstmögliche Wert.
'''

#tfds.show_examples(train , info)  # Anzeigen einiger Samples mit entsprechendem Label
#tfds.as_dataframe(train.take(11), info)  # Anzeigen von 21 Samples als Datenframe, funktioniert anscheinend nicht, da Pandas glaube ich nicht installiert ist



# - Aufgabe 2.2 Setting up the data pipeline -------------------------------------
# Zuerst wird ein Inputvektor benötigt, welcher das Bild darstellt. Außerdem wird ein Outputvektor gebraucht, welcher das Target darstellen soll.
# Jedes Sample wird als (pixels, target) Tupel dargestellt
# Da ein Neuronales Netz aber nicht mit 2D, 3D oder höherdimensionalen Daten umgehen kann, muss das Bild auf 1D reduziert werden.
# Die map Funktion nimmt ein Inputelement und bildet es entsprechend der angegebenen Lambda Funktion auf ein Outputelement ab.
# In Python können Lambda Funktionen genutzt werden, um Funktionen on the fly und ohne Namen zu definieren.
def prepare_mnist_data(data):
    # BEACHTE: jeder Schritt kann als eine Maschine in einer Produktionsfirma angesehen werden.
    # Die Firma macht NICHT eine Woche lang Schritt 1, eine Woche lang Schritt 2, ...
    # Es wird also nicht darauf gewartet, dass alle Samples Schritt 1 durchlaufen haben.
    # Sondern es wird ein Stream über alle Schritte gebildet. Wie ein Laufband, an welchem nebendran die Maschinen stehen und
    # ein Produkt nach dem anderen an die nächste Maschine weitergeben wird.
    # Dies ist besonders bei großen Datensätzen, die nicht auf einmal in den Speicher passen, extrem nützlich.

    # - Maschine 1 -----
    # Flatten des 2D Bildes in einen Vektor (letzte Dimension kann ignoriert werden, da sie gleich 1 ist)
    # lambda eingabeparameter: returnvalue
    # Beachte, dass der returnvalue hier ein Tupel aus (image, target) ist.
    # Die tf.reshape Funktion erhält als Eingabeparameter einen Tensor und eine Shape, welche hier (-1, ) ist.
    # Wenn eine Komponente von shape den speziellen Wert -1 hat, wird die Größe dieser Dimension so berechnet, dass die Gesamtgröße konstant bleibt.
    # Insbesondere wird eine Shape mit dem Wert [-1] zu 1-D abgeflacht. Höchstens eine Komponente von shape kann -1 sein. => Bild = 1D Inputvektor!
    data = data.map(lambda image, target: (tf.reshape(image, (-1, )), target))
    
    # - Maschine 2 -----
    # Umwandlung der Pixelwerte von uint8 in float32 mittels Casting
    data = data.map(lambda image, target: (tf.cast(image, tf.float32), target))

    # - Maschine 3 -----
    # Reskalierung der Daten, da wir in unserem Netzwerk hohe Werte auf der einen Seite der null nicht gut finden.
    # Wir wollen, dass die Werte um 0 herum zentriert sind und kleine Werte zwischen -1 und 1 annehmen.
    # => Normalization, Normalverteilung um 0 mit mean = 0 und sigma = 1.
    # Dafür müssen wir die Pixelwerte durch 256 / 2 = 128 teilen, wodurch wir Werte zwichen 0 und 2 erreichen
    # und anschließend minus 1 rechnen, um Werte zwischen - 1 und 1 zu bekommen.
    data = data.map(lambda image, target: ((image / 128.0) - 1.0, target))

    # - Maschine 4 -----
    # Bei einer Klassifikation sollte das Target keine Zahl sondern ein One Hot Vektor sein.
    # Der Parameter depth entspricht der Anzahl an Klassen. 
    # Hier haben wir die Klassen Null, Eins, Zwei, Drei, Vier, Fünf, Sechs, Sieben, Acht und Neun.
    data = data.map(lambda image, target: (image, tf.one_hot(target, depth = 10)))

    # - Maschine 5 -----
    # Damit wir nicht bei jedem erneuten Durchlauf/Zugriff auf die Daten alle Schritte nochmal durchlaufen müssen,
    # speichern wir die Ergebnisse der bisherigen Schritte im Cache. Dabei nehmen wir an, dass das
    # MNIST Datenset klein genug ist, um es im Cache (Memory) zu speichern.
    data = data.cache()

    # - Maschine 6 -----
    # Die shuffle Funktion mischt die Reihenfolge der Daten durcheinander, da wir in jedem Durchlauf gerne die Daten in einer anderen
    # Reihenfolge hätten. 
    data = data.shuffle(1000)

    # - Maschine 7 -----
    # Die batch Funktion konkateniert 32 Samples entlang der ersten Achse hintereinander. => batchsize (hier also 32 Bilder mit je 28x28 Pixeln)
    data = data.batch(32)

    # - Maschine 8 -----
    # Die prefetch Funktion sorgt dafür, das mindestens immer 20 Samples zur Benutzung bereit stehen.
    # Solange keine 20 Samples bereit sind, versucht die Firma so schnell wie möglich zu laufen, bis 20 ready to use Produkte bereit stehen.
    # Da die GPU die Daten schneller durch das ANN jagt, als die CPU die Daten preprocessen kann, 
    # sollten die beiden Prozesse über den Fetch Befehl parallelisiert werden.
    data = data.prefetch(20)

    # Das preprocesste Datenset zurückgeben.
    # - Maschine 9 -----
    return data


# Preprocessing Methode sowohl auf den Trainingsdatensatz als auch auf den Testdatensatz anwenden.
# Die Funktion apply kann auf einem Tensorflow.Dataset.Object aufgerufen werden und wendet die angegebene Methode auf das Datenset an.
train_dataset = train.apply(prepare_mnist_data)
test_dataset = test.apply(prepare_mnist_data)





