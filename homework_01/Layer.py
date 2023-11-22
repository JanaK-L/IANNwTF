import numpy as np
import AktivierungsUndLossFunktionen as funks
import sys

# - Aufgabe 2.4 -------------------------------------------------------------------------------
class Layer():

    # Konstruktor erwartet:
    # - die Aktivierungsfunktion, wenn true Sigmoid, wenn false Softmax
    # - Anzahl an Neuronen (Perceptrons) in diesem Layer = anzahlNeurons
    # - Anzahl an Inputs (Anzahl an Neuronen im vorherigen Layer) = inputSize, wobei hier die extra Zeile für den Bias noch nicht enthalten ist
    def __init__(self, sigmoid:bool, anzahlNeurons:int, inputSize:tuple):
        self.sigmoid = sigmoid
        self.anzahlNeuronsVorherigerLayer = inputSize[1]  # Anzahl Zeilen der Gewichtsmatrix
        self.anzahlNeurons = anzahlNeurons  # Anzahl Spalten der Gewichtsmatrix
        
        # Klassenattribut für die Gewichtsmatrix (inklusive Bias -> eine Zeile mehr wird für den Bias benötigt, also + 1) anlegen.
        # Die Gewichtsmatrix wird mit kleinen Random Werten aus einer Normalverteilung mit µ = 0 und sigma = 0.2 initialisiert.
        mü = 0 # Durchschnitt = μ
        sigma = 0.2 # Standardabweichung = σ
        self.W = np.random.normal(mü, sigma, (self.anzahlNeuronsVorherigerLayer + 1, self.anzahlNeurons))  # Tupel mit (Zeile, Spalte) angeben, um die Größe des Arrays zu bestimmen
        # Der Bias wird mit 0 initialisiert.
        # print(self.W[inputSize - 1])  # -1, da Arrayindex bei 0 anfängt
        for i in range(self.anzahlNeurons):
            self.W[self.anzahlNeuronsVorherigerLayer][i] = 0
        # print("Gewichtsmatrix:\n", self.W)
        # Slicing verwenden, um die letzte Zeile (=> -1) der Matrix wegzulassen
        self.WohneBias = self.W[0:-1, 0:]
        #print("\nmitBias", self.W, "\nohneBias", self.WohneBias)
        #print("---------")

        # Aktivierungsfunktionsobjekt erzeugen
        aktivierungsfunktion = None
        if self.sigmoid:
            self.aktivierungsfunktion = funks.Sigmoid()
        else:
            self.aktivierungsfunktion = funks.Softmax() 

        self.preAktivierung = None  # Preactivation = d
        self.activation = None  # Activation = a
        self.delta = None  # ist der Gradient für den Bias
        self.gradient = None  # ist der Gradient für die Gewichtsmatrix
        


    def __str__(self):
        return "Anzahl Neuronen des vorherigen Layers: " + str(self.anzahlNeuronsVorherigerLayer) + ", Anzahl Neuronen: " + str(self.anzahlNeurons) + ", sigmoid? " + str(self.sigmoid)


    # Forward Funktion  
    def forward(self, input:np.ndarray, minibatchsize:int):
        # input muss die shape (minibatchsize, inputSize) haben
        expectedShape = (minibatchsize, self.anzahlNeuronsVorherigerLayer)
        #print("erwartet", expectedShape)
        #print("erhalten", input.shape)
        if input.shape == expectedShape:
            # Während die Gewichtsmatrix bereits die extra Zeile für den Bias beinhaltet, fehlt die extra Zeile für den Bias beim Input
            inputMitBias = self.addNeutralesElement(input)

            # Preactivation berechnen (inputs * Gewichtsmatrix) inklusive Bias
            #print(inputMitBias.shape, self.W.shape)
            self.preAktivierung = inputMitBias @ self.W
            #print("preactivierung Hinweg", self.preAktivierung.shape)
            # Ergebnis des Vorwärtsschritts speichern und zurückgeben
            self.activation = self.aktivierungsfunktion.call(self.preAktivierung, minibatchsize, self.anzahlNeurons)
            return self.activation
        else:
            print("Fehler! Die Shape der inputs passt bei der forward() Funktion nicht!")
            sys.exit(-1)  # Fehler, Exitcode 0 wäre kein Fehler



    # Backpropagation Step
    def backward(self, activationVorherigerLayer:np.ndarray, klassifikationsErgebnisDesML:np.ndarray, deltaNachfolgenderLayer:np.ndarray, WohneBiasVomNachfolgendenLayer:np.ndarray):
        # bei dem Outputlayer wird die Softmaxaktivierungsfunktion in Kombination mit der CCE Loss Funktion verwendet
        if not self.sigmoid:
            # delta berechnen, wobei deltaNachfolgenderLayer hier die targets enthält
            self.delta = self.aktivierungsfunktion.backwards(klassifikationsErgebnisDesML, deltaNachfolgenderLayer)
            self.delta = self.delta.astype("float32")
            # Gradient berechnen 
            self.gradient = activationVorherigerLayer.transpose() @ self.delta
            self.gradient = self.gradient.astype("float32")
            #print("delta:\n", self.delta, self.delta.shape, "\nGradient", self.gradient, self.gradient.shape)
        else:
            # delta berechnen
            self.delta = deltaNachfolgenderLayer @ WohneBiasVomNachfolgendenLayer.transpose()
            self.delta = self.delta * self.aktivierungsfunktion.backwards(self.preAktivierung)  # Element weise Multiplikation
            self.delta = self.delta.astype("float32")
            # Gradient berechnen
            self.gradient = activationVorherigerLayer.transpose() @ self.delta
            self.gradient = self.gradient.astype("float32")



    # Funktion zum Updaten der Gewichtsmatrix und dem Bias mit Hilfe von Delta und Gradient Attributen des Layers
    def update(self, lernrate:float):
        # Bias updaten: gradientVomBias = delta, wobei delta die shape [batchsize, AnzahlNeuronenImLayer] hat
        # während der Bias die shape [1, AnzahlNeuronenImLayer] hat
        # Aufgrund der unterschiedlichen Shapes wird an dieser Stelle der mittlere Gradient für den Bias verwendet
        # Mittelwert über die Batch Dimension (axis = 0, da batchsize die erste Achse definiert) nehmen
        biasGradient = np.average(self.delta, axis = 0) # jede spalte wird aufsummiert und der Mittelwert der Spalte wird gebildet
        # Bias aus self.W extrahieren, letzte Zeile und alle Spalten Einträge der letzten Zeile
        biasAlt = self.W[-1:, 0:].flatten()
        biasNeu = biasAlt - (biasGradient * lernrate)
        
        # neue Gewichte (ohne Bias) = alte Gewichte - Gradient * Lernrate
        self.WohneBias = self.WohneBias - (self.gradient * lernrate)

        # self.W updaten
        for i in range(self.anzahlNeurons):
            self.W[self.anzahlNeuronsVorherigerLayer][i] = biasNeu[i]

        for i in range(self.WohneBias.shape[0]):
            for j in range(self.WohneBias.shape[1]):
                self.W[i][j] = self.WohneBias[i][j]

        # Alles wieder zurücksetzen
        self.preAktivierung = None  # Preactivation = d
        self.activation = None  # Activation = a
        self.delta = None  # ist der Gradient für den Bias
        self.gradient = None  # ist der Gradient für die Gewichtsmatrix




    # Inputmatrix mit einer Spalte neutraler Elemente für den Bias erweitern (1en einfügen)
    def addNeutralesElement(self, input):
        # erzeuge eine Matrix voll mit 1en und überschreibe alle Werte mit den Werten in input bis auf die rechte Spalte
        inputMatrixMitNeutralemElement = np.ones((input.shape[0], input.shape[1] + 1))
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                inputMatrixMitNeutralemElement[i][j] = input[i][j]
        # print("inputMatrixMitNeutralemElement\n", inputMatrixMitNeutralemElement)
        return inputMatrixMitNeutralemElement
    

     