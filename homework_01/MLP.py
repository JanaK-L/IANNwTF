import numpy as np
import Layer

# - Aufgabe 2.5 ----------------------------------------------------------------
class MLP():
    # - Der initiale Input des MLP hat die Shape (minibatchsize, 64)
    # - Der Output des MLP hat die Shape (minibatchsize, 10)

    # Größe des Arrays sizeOfEachLayer ergibt die Anzahl an Layern und die Werte in dem Array selbst,
    # sagen aus, wie viele Neuronen in dem jeweiligen Layer enthalten sein sollen.
    def __init__(self, sizeOfEachLayer:np.ndarray, minibatchsize:int, lernrate:float):
        self.layers = []  # leere Liste
        self.anzahlLayer = len(sizeOfEachLayer)
        self.minibatchsize = minibatchsize
        self.initialeEingabeMatrix = None
        self.lernrate = lernrate
     
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

        print("-------------------------------------------------")

    # Funktion, um den Hinweg im MLP zu starten        
    def hinweg(self, initialeEingabeMatrix) -> np.ndarray:
        self.initialeEingabeMatrix = initialeEingabeMatrix
        vorwärts = initialeEingabeMatrix
        # Aufrufen der forward Funktion eines jeden Layers mit anschließender Weitergabe der Ergebnisse an den nachfolgenden Layer
        for i in range(self.anzahlLayer):
            vorwärts = self.layers[i].forward(vorwärts, self.minibatchsize)
            # print("Ergebnis von Layer", i, "hat die shape", vorwärts.shape, "\n", vorwärts, "\n")

        return vorwärts


    # Funktion, um den Rückweg im MLP zu starten
    def rückweg(self, klassifikationsErgebnisDesMLP:np.ndarray, targets:np.ndarray) -> np.ndarray:
        # Backpropagationschritt für den letzten Layer berechnen
        self.layers[self.anzahlLayer - 1].backward(self.layers[self.anzahlLayer - 2].activation, klassifikationsErgebnisDesMLP, targets, None)
        # start, stop, step (-1 = rückwärts durchlaufen)
        for i in range(self.anzahlLayer - 2, 0, -1):
            #print(i)
            #  activationVorherigerLayer, klassifikationsErgebnisDesML, deltaNachfolgenderLayer, WohneBiasVomNachfolgendenLayer
            self.layers[i].backward(self.layers[i - 1].activation, None, self.layers[i + 1].delta, self.layers[i + 1].WohneBias)
            #print(self.layers[i])
        
        # Backpropagationschritt für den ersten Layer berechnen: Input des Netzwerkes
        #print(self.initialeEingabeMatrix.shape, self.layers[1].delta.shape, self.layers[1].WohneBias.shape)
        self.layers[0].backward(self.initialeEingabeMatrix, None, self.layers[1].delta, self.layers[1].WohneBias)



    def update(self):
        # Gewichtematrix und Bias eines jeden Layers updaten
        for i in range(self.anzahlLayer):
            self.layers[i].update(self.lernrate)

