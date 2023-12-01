import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
# from google.colab import files  # wird nur benötigt, wenn wir in Colab unsere Graphen downloaden wollen

# - Aufgabe 2.1 Laden des Datensets -------------------------------------
(test, train), info = tfds.load("mnist", split=["train", "test"], as_supervised = True, with_info = True)  # Test und Training Samples als Tuples (features, label) laden

#print("\n\nInformationen zum Datenset:\n", info)

'''
• How many training/test images are there?
  Test = 10.000 Samples
  Training = 60.000 Samples
• What’s the image shape?
  Die Bilder haben eine Shape von (28, 28, 1), wobei die Bilder 28x28 = 784 Pixel beinhalten
  und die 1 dafür steht, dass die Bilder nur über einen Kanal (Grauwerte) verfügen. Panchromatisch vermutlich?
• What range are pixel values in?
  Der dtype der Images ist uint8 (8 Bit unsigned Integer). Demnach ist 0 der kleinste und 255 der höchstmögliche Wert.
'''

#tfds.show_examples(train , info)  # Anzeigen einiger Samples mit entsprechendem Label
#tfds.as_dataframe(train.take(11), info)  # Anzeigen von 11 Samples als Datenframe



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
    # Insbesondere wird eine Shape mit dem Wert [-1] zu 1D abgeflacht. Höchstens eine Komponente von shape kann -1 sein. => Bild = 1D Inputvektor!
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



# - Aufgabe 2.3 Aufbau des Netzwerkmodells -------------------------------------
# Das Modell erbt von der Klasse tf.keras.Model
class MNIST_Modell(tf.keras.Model):

  # Konstruktor Funktion
  def __init__(self, layer_sizes: list, output_size: int = 10):
    # Aufruf des Konstruktors der Superklasse
    super(MNIST_Modell, self).__init__()

    # Den nachfolgenden Codeabschnitt könnte man via List Comprehension auch in einer Zeile schreiben
    # Hidden Layer erzeugen basierend auf der Liste der layer_sizes
    self.layersListe = []  # leere Liste erstellen, in welcher die Layer gespeichert werden sollen
    for size in layer_sizes:
      # Dense steht für densely connected Layer und bedeutet, dass jedes Neuron von linkem
      # Layer mit jedem Neuron im rechten Layer verbuden ist
      newLayer = tf.keras.layers.Dense(units = size, activation = tf.nn.sigmoid)  # anstatt tf.nn.sigmoid kann man auch "sigmoid" als String angeben
      # erzeugten Layer in die Liste von Layern eintragen
      self.layersListe.append(newLayer)

    # Outputlayer erzeugen
    # Der Outputlayer hat so viele units, wie Klassen vorhanden sind, bei MNIST also 10
    # Da wir wieder Klassifizieren, benötigen wir die Softmax Funktion im Outputlayer
    self.output_layer = tf.keras.layers.Dense(units = output_size, activation = tf.nn.softmax)


  # Feed Forward Funktion: Berechnung des Hinwegs, wobei das Ergebnis eines Layers
  # an den nachfolgenden Layer weitergegeben wird.
  # Der Übergabeparameter x steht für die Inputwerte.
  def call(self, x):
    for layer in self.layersListe:  # Hidden Layers
      x = layer(x)
    y = self.output_layer(x)  # Outputlayer
    return y  # y = die Prediction des Modells



# - Aufgabe 2.4.1 Trainingsfunktion -------------------------------------
# Trainiert das Modell basierend auf einem einzelnen Input Target Paar.
# (model ist etwas, das callable ist, wie z.b. eine Funktion, oder ein Objekt, das über eine call() Funktion verfügt)
# loss_function muss ebenfalls callable sein
def train_step(model, input, target, loss_function, optimizer):
  # berechne den Loss zwischen den Targets und den vorhergesagten Werten innerhalb des Context Managers
  with tf.GradientTape() as tape:
    # Aufruf der call bzw. der __call__ Funktion, welche das Ergebnis des Hinweges, also die Prediction, zurück gibt
    prediction = model(input)
    # den Losswert für die berechnete Prediction auf Basis der Targets bestimmen
    loss = loss_function(target, prediction)

  # Berechnung der Gradienten mit Respekt auf model.trainable_variables
  gradients = tape.gradient(loss, model.trainable_variables)

  # Die Funktion apply_gradients() erhält das GradientTape und die Liste von zu updatendend Variablen des Modells
  # es ist wichtig, dass die Anzahl an Elementen in dem GradientTape der Anzahl an Elementen in der Parameter_estimate Liste die selbe Anzahl ist
  # Updaten der model.trainable_variables (Gewichte und Biase) mit Hilfe der zuvor berechneten Gradienten
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  # Loss zurück geben, damit wir ihn plotten können
  return loss




# - Aufgabe 2.4.2 Testingfunktion -------------------------------------
# Testet das Modell basierend auf Daten, die das Modell vorher noch nicht gesehen hat (hier dürfen keine Trainingsdaten verwendet werden)!
def test(model, test_data, loss_function):
  # Achtung: Python List Kram ist ineffizient, später sehen wir, wie es besser in TF Code geht.
  # Accuracy = war die Vorhersage richtig oder falsch
  test_accuracy_aggregator = []  # leere Liste für die Accuracywerte
  # Loss = kontinuierlicher Wert, der aussagt, wie weit wir von richtigem Wert entfernt sind
  test_loss_aggregator = []  # leere Liste für die Losswerte

  # iterieren über das Dataset
  for (input, target) in test_data:
    # BEACHTE: Beim Testen werden die Parameter des Modells nicht geupdatet. Es werden keine Gradienten berechnet!
    # Daher wird nur der Hinweg, also die Prediction berechnet!!!
    prediction = model(input)

    # berechne den Loss
    sample_test_loss = loss_function(target, prediction)

    # Prüfe, ob der Index des größten Wertes in unserer Prediction dem One Hot Target Index mit der 1 entspricht
    sample_test_accuracy = np.argmax(target, axis = 1) == np.argmax(prediction, axis = 1)
    # Da wir Mini Batching verwenden, müssen wir den Durchschnitt bilden
    sample_test_accuracy = np.mean(sample_test_accuracy)

    # aktuellen Loss und Accuracy in unsere Listen einfügen
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  # Durchschnitt für unsere Listen berechnen
  test_loss = tf.reduce_mean(test_loss_aggregator)
  test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

  # Gesamtloss und Gesamtgenauigkeit zurückgeben
  return test_loss, test_accuracy




# - Aufgabe 2.5.1 Visualisierungsfunktion der Performanz des Modells -------------------------------------
# Der Code zu dieser Aufgabe stammt vom Aufgabenblatt selbst.
def visualization(train_losses, test_losses, test_accuracies):
  """ Visualizes accuracy and loss for training and test data using the mean of each epoch.
  Loss is displayed in a regular line, accuracy in a dotted line.
  Training data is displayed in blue, test data in red.

  Parameters
  ----------
  train_losses :numpy.ndarray training losses
  train_accuracies: numpy.ndarray training accuracies
  test_losses: numpy.ndarray test losses
  test_accuracies: numpy.ndarray test accuracies
  """

  plt.figure()
  line1, = plt.plot(train_losses, "b-")
  line2, = plt.plot(test_losses, "r-")
  line4, = plt.plot(test_accuracies, "r:")
  plt.xlabel("Epochs")
  plt.ylabel("Loss / Accuracy")
  plt.title("Initial_Setup")
  plt.legend((line1 , line2 , line4), (" training loss ", " test loss ", " test accuracy "))
  plt.savefig("Initial_Setup.png")
  # files.download("Initial_Setup.png")  # Downloaden des Graphen
  plt.show()




# - Aufgabe 2.5.2 Messung der Performanz des Modells -------------------------------------
# Hyperparameter
anzahl_epochs = 10
lernrate = 0.1

# Initialisierung des Modells mit 2 Hidden Layers mit je 256 Units und einem Outputlayer der Größe 10
model = MNIST_Modell(layer_sizes = [256, 256], output_size = 10)

# Initialisierung der CCE als Loss Funktion
cce = tf.keras.losses.CategoricalCrossentropy()

# Initialisierung des Optimizers, SGD steht für Stochastic Gradient Descent
optimizer = tf.keras.optimizers.legacy.SGD(lernrate)

# Leere Listen für die spätere Visualisierung erstellen
train_losses = []
test_losses = []
test_accuracies = []

# Testen des Modells vor dem Training sowohl auf dem Trainings- als auch auf dem Testdatenset
test_loss, test_accuracy = test(model = model, test_data = test_dataset, loss_function = cce)
train_loss, _ = test(model = model, test_data = train_dataset, loss_function = cce)  # Genauigkeit auf dem Trainingsdatenset wird nicht benötigt

# Vermerken der Ergebnisse der Tests vor dem Training in den Listen
train_losses.append(train_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)


# START DES TRAININGS!!!!!!!!!!!!!
for epoch in range(anzahl_epochs):  # in einem Epoch wird das Modell EINMAL auf dem kompletten Trainingsdatenset trainiert
  print("starte epoch", epoch)
  # leere Liste erstellen, in welcher die Losses des aktuellen Epochs gesammelt werden
  epoch_loss_aggregator = []

  # Iteriere über das gesamte Datenset und berechne und merke den Loss für jedes Sample im Datenset
  for input, target in train_dataset:
    train_loss = train_step(model = model, input = input, target = target, loss_function = cce, optimizer = optimizer)
    epoch_loss_aggregator.append(train_loss)

  # Füge den Durchschnitt der Losswerte des aktuellen Epochs in die Trainings Loss Liste hinzu
  train_losses.append(tf.reduce_mean(epoch_loss_aggregator))

  # Teste, wie gut die Performanz des Modells nach jedem Epoch ist und füge die Werte den Listen hinzu
  test_loss, test_accuracy = test(model = model, test_data = test_dataset, loss_function = cce)
  test_losses.append(test_loss)
  test_accuracies.append(test_accuracy)


# Plotten der Performanzergebnisse
visualization(train_losses = train_losses, test_losses = test_losses, test_accuracies = test_accuracies)

# Beachte: Normalerweise ist der Training Loss kleiner als der Testing Loss. Da wir während des Trainings bereits den
# TrainingsLoss aufakkummuliert haben und nicht erst am Ende, hat der Training Loss on Average ein halbes Epoch
# mehr an Updates gesehen als der TestLoss. Wenn wir den Trainingsloss und den Testloss beide am Ende eines Epochs
# berechnen würden, wäre der Training Loss kleiner als der Testloss.




# - Aufgabe 3 Hyperparameter -------------------------------------
# Lernrate, batchsize, Anzahl und Größe der Layer, Optimizer (and e.g. in SGD’s case the momentum Hyperparameter)

# Runde 1 (Initial Setup):
# Epochs = 10
# Lernrate = 0,1
# Batchsize = 32
# 2 Hidden Layer mit je 256 Units
# Optimizer SGD mit Momentum = 0

# Runde 2 (Hohe Lernrate):
# Epochs = 10
# Lernrate = 0,5
# Batchsize = 32
# 2 Hidden Layer mit je 256 Units
# Optimizer SGD mit Momentum = 0

# Runde 3 (Mehr Layers):
# Epochs = 5
# Lernrate = 0,1
# Batchsize = 32
# 4 Hidden Layer mit je 256 Units
# Optimizer SGD mit Momentum = 0

# Runde 4 (Mehr Units):
# Epochs = 10
# Lernrate = 0,1
# Batchsize = 32
# 2 Hidden Layer mit je 512 Units
# Optimizer SGD mit Momentum = 0

# Runde 5 (Hohe Batchsize):
# Epochs = 10
# Lernrate = 0,1
# Batchsize = 128
# 2 Hidden Layer mit je 256 Units
# Optimizer SGD mit Momentum = 0



"""
Interpretation der Ergebnisse
Beim initialen Setup sinkt der Trainings- und der Testloss in den ersten zwei Epochen am stärksten und die Genauigkeit steigt an.
In Epoche 5 ist beim Testloss ein minimaler Anstieg zu sehen. Danach sinken der Test- und Trainingsloss weiterhin leicht ab und die Genauigkeit steigt leicht.

Im Vergleich dazu ist bei der hohen Lernrate ein minimaler Anstieg in des Testlosses in Epoche 3 und ein hoher Anstieg des Testlosses in Epoche 5 zu sehen.
Zudem ist in Epoche 5 auch ein deutlicher Abfall in der Genauigkeit zu sehen. Dies liegt vermutlich daran, dass aufgrund der hohen Lernrate beim Gradient
Descent ein zu großer Schritt gemacht und über das Optimum hinüber geschritten wird.

Bei einer höheren Batchsize fällt auf, dass die Genauigkeit etwas langsamer ansteigt und der Trainings und Testloss langsamer fallen.
Dies könnte daran liegen, dass bei einer größeren Batchsize das Netzwerk pro Epoch nicht so oft geupdated wird, wie bei einer kleineren Batchsize.
Dementsprechend benötigt das Netzwerk ein bis zwei Durchläufe mehr, um die Klassifikation zu lernen.

Bei mehr Layern fällt auf, dass die Genauigkeit sehr langsam steigt und der Trainings- und Testloss sehr langsam fallen. Mehr Layer scheinen das Netzwerk
beim Trainieren eher zu verwirren, als zu helfen.

Im Vergleich zum initialen Setup ist fast kein Unterschied zu sehen. Vermutlich sind 256 Units pro Layer bereits ausreichend, um die Klassifikation zu lernen,
sodass die Hinzunahme weiterer Units das Ergebnis nicht sichtlich verbessern.
"""