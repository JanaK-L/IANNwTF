import numpy as np
import Layer

# - Aufgabe 2.5 ----------------------------------------------------------------
class MLP():
    # - Der initiale Input des MLP hat die Shape (minibatchsize, 64)
    # - Der Output des MLP hat die Shape (minibatchsize, 10)

    # Größe des Arrays sizeOfEachLayer ergibt die Anzahl an Layern und die Werte in dem Array selbst,
    # sagen aus, wie viele Neuronen in dem jeweiligen Layer enthalten sein sollen.
    def __init__(self, sizeOfEachLayer:np.ndarray, minibatchsize:int):
        self.layers = []  # leere Liste
        self.anzahlLayer = len(sizeOfEachLayer)
        self.minibatchsize = minibatchsize
     
        # Inputlayer hinzufügen
        anzahlPixel = 64  # 8 x 8 Pixel
        inputlayer = Layer.Layer(True, sizeOfEachLayer[0], (self.minibatchsize, anzahlPixel))
        self.layers.append(inputlayer)

        # entsprechend viele Layer -1 mit der Sigmoiden Aktivierungsfunktion erzeugen
        for i in range(1, self.anzahlLayer - 1):
            # (Sigmoid, anzahlNeuronenAktuellerLayer, anzahlNeuronenVorherigerLayer)
            newLayer = Layer.Layer(True, sizeOfEachLayer[i], (self.minibatchsize, sizeOfEachLayer[i -1]))
            self.layers.append(newLayer)

        # Outputlayer mit Softmax Funktion hinzufügen
        outputlayer = Layer.Layer(False, sizeOfEachLayer[self.anzahlLayer - 1], (self.minibatchsize, sizeOfEachLayer[self.anzahlLayer - 2]))
        self.layers.append(outputlayer)

        for i in self.layers:
            print(i)


    # Funktion, um den Hinweg im MLP zu starten        
    def hinweg(self, initialeEingabeMatrix) -> np.ndarray:
        vorwärts = initialeEingabeMatrix
        # Aufrufen der forward Funktion eines jeden Layers mit anschließender Weitergabe der Ergebnisse an den nachfolgenden Layer
        for i in range(self.anzahlLayer):
            vorwärts = self.layers[i].forward(vorwärts, self.minibatchsize)
            print("Ergebnis von Layer", i, "hat die shape", vorwärts.shape, "\n", vorwärts, "\n")

        return vorwärts


    # Funktion, um den Rückweg im MLP zu starten
    def rückweg(self, klassifikationsErgebnisDesMLP:np.ndarray, targets:np.ndarray) -> np.ndarray:
        # start, stop, step (-1 = rückwärts durchlaufen)
        for i in range(self.anzahlLayer - 1, -1, -1):
            print(i)

        return None

