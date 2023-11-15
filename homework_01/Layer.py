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
        for i in range(anzahlNeurons):
            self.W[self.anzahlNeuronsVorherigerLayer][i] = 0
        # print("Gewichtsmatrix:\n", self.W)
        self.delta = None
        self.gradient = None
        

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
            preAktivierung = inputMitBias @ self.W

            # Aktivierungsfunktionsobjekt erzeugen
            aktivierungsfunktion = None
            if self.sigmoid:
                aktivierungsfunktion = funks.Sigmoid()
            else:
                aktivierungsfunktion = funks.Softmax()  
            
            aktivierung = aktivierungsfunktion.call(preAktivierung, minibatchsize, self.anzahlNeurons)
            # Ergebnis des Vorwärtsschritts zurückgeben
            return aktivierung
        else:
            print("Fehler! Die Shape der inputs passt bei der forward() Funktion nicht!")
            sys.exit(-1)  # Fehler, Exitcode 0 wäre kein Fehler


    # Backpropagation Step
    def backward(self, activation:np.ndarray, targets:np.ndarray):
        # bei dem Outputlayer wird die Softmaxaktivierungsfunktion in Kombination mit der CCE Loss Funktion verwendet
        if not self.sigmoid:
            # delta berechnen
            self.delta = funks.CCE().backwards(activation, targets)
            # Gradient berechnen 
            self.gradient = activation.transpose() @ self.delta
        else:
            pass


    # Inputmatrix mit einer Spalte neutraler Elemente für den Bias erweitern (1en einfügen)
    def addNeutralesElement(self, input):
        # erzeuge eine Matrix voll mit 1en und überschreibe alle Werte mit den Werten in input bis auf die rechte Spalte
        inputMatrixMitNeutralemElement = np.ones((input.shape[0], input.shape[1] + 1))
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                inputMatrixMitNeutralemElement[i][j] = input[i][j]
        # print("inputMatrixMitNeutralemElement\n", inputMatrixMitNeutralemElement)
        return inputMatrixMitNeutralemElement
    

     