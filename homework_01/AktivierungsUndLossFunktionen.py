import math
import numpy as np
import sys


# - Aufgabe 2.2 -------------------------------------------------------------------------------
class Sigmoid():

    # Konstruktor
    def __init__(self):
        pass

    # Funktion zur Berechnung der Aktivierung erwartet ndarrays der shape (minibatchsize, anzahlNeurons)
    # wobei anzahlNeurons die Anzahl an Perceptrons in dem Layer, in welchem die Aktivierungsfunktion aufgerufen wird, ist.
    def call(self, preactivation:np.ndarray, minibatchsize:int, anzahlNeurons:int) -> np.ndarray:
        # Überprüfe, ob das preActivation Array die richtige Form hat (minibatchsize, 64)
        expectedShape = (minibatchsize, anzahlNeurons)  # + 1 für den Bias
        #print("erwartet:", expectedShape)
        #print("erhalten", preactivation.shape)
        if preactivation.shape == expectedShape:
            self.preactivation = preactivation  # d steht für die preactivation
            # Eigentliche Berechnung der Sigmoiden Funktion durchführen
            return 1 / (1 + math.e **(-preactivation))
        else:
            print("Fehler! Shape der preactivation entspricht nicht der für die Sigmoide Funktion erwarteten Shape!")    
            sys.exit(-1)  # Fehler, Exitcode 0 wäre kein Fehler


    # Erste Ableitung der sigmoiden Funktion berechnen    
    def backwards(self, preactivation:np.ndarray) -> np.ndarray:
        return (1 / (1 + math.e **(-preactivation))) * (1 - (1 / (1 + math.e **(-preactivation))))








# - Aufgabe 2.3 -------------------------------------------------------------------------------
class Softmax():

    # Konstruktor
    def __init__(self):
        pass


    # Funktion zur Berechnung der Aktivierung
    # Die Softmaxfunktion wandelt den Eingabevektor in eine Wahrscheinlichkeitsverteilung um, d. h. der Eingabevektor ist an jedem Eintrag positiv und summiert sich zu eins auf.
    # Das Ergebnis dieser Aktivierung sind also die Wahrscheinlichkeiten dafür, dass ein Graubild eine der Zahlen 0,1,2,3,4,5,6,7,8 oder 9 darstellt. Klassifikation des Graubildes.
    def call(self, preactivation:np.ndarray, minibatchsize:int, anzahlNeurons:int = 10) -> np.ndarray:
        # Überprüfe, ob das preActivation Array die richtige Form hat (minibatchsize, 10)
        expectedShape = (minibatchsize, anzahlNeurons)
        #print("erwartet:", expectedShape)
        #print("erhalten", preactivation.shape)
        if preactivation.shape == expectedShape:    # d steht für die preactivation
            zähler = math.e **preactivation
            # Zeile für Zeile nur aus den Werten in der Zeile eine Summe bilden (axis = 1), Dimension beibehalten, damit ein Spaltenvektor erhalten bleibt, anstatt einem Zeilenvektor
            nenner = np.sum(zähler, axis = 1, keepdims = True)
            ergebnis = zähler.copy()
            for i in range(zähler.shape[0]):
                for j in range(zähler.shape[1]):
                    ergebnis[i][j] = zähler[i][j] / nenner[i]
                # print("next row")
           
            return ergebnis
        
        else:
            print("Fehler! Shape der preactivation entspricht nicht der für die Softmax Funktion erwarteten Shape!")    
            sys.exit(-1)  # Fehler, Exitcode 0 wäre kein Fehler


    # kombinierte erste Ableitung aus CCE und Softmax
    def backwards(self, klassifikationsErgebnisDesMLP: int, targets:np.ndarray) -> np.ndarray:
        # erste Ableitung von CCE und Softmax zusammen ist: a - t = aktivierung - targetwert
        # klassifikazionsErgebnisDesMLP muss shape (minibatchsize, 10) haben
        # shape sollte (minibatchsize, 10) sein
        return klassifikationsErgebnisDesMLP - targets    
       

        







# - Aufgabe 2.6 -------------------------------------------------------------------------------
# CCE = categorical cross-entropy loss function
class CCE():

    # Konstruktor
    def __init__(self):
        pass


    def call(self, klassifikationsErgebnisDesMLP: int, targets:np.ndarray) -> np.ndarray:
        # Die Targets liegen in der One Hot Codierung mit einer shape von [minibatchsize, 10] vor.
        # Das KlassifikationsErgebnisDesMLP (= prediction) liegt ebenfalls in der shape [minibatchsize, 10] vor.
        innen = targets * np.log(klassifikationsErgebnisDesMLP)
        cceLoss = -1 * np.sum(innen, axis = 1)  # Lossmatrix hat die shape [minibatchsize, 1], da für jedes Bild im Minibatch ein Losswert existiert
        return cceLoss                              


    # wird nicht benötigt, da die kombinierte Ableitung aus CCE und Softmax bereits in der Softmax backwards Funktion berücksichtigt wird    
    #def backwards(self):
       #pass