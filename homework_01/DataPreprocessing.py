from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle


def preProcessing():
    # - 1.) Laden des Datensets ---------------------------------------
    # Die Funktion load_digits() liefert unter Angabe des Parameters return_X_y = True ein Tupel aus zwei ndarrays zurück.
    # Das erste ndarray enthält ein 2D-ndarray der Shape (1797, 64), wobei jede Zeile ein Sample (eine Probe) und jede Spalte die Features (Merkmale) darstellt.
    # Das erste ndarray enthält demnach die 1797 Graubilder des Datensets, wobei jedes Graubild aus 8x8 Pixeln, also aus 64 Grauwerten besteht. 
    # Das zweite ndarray der Shape (1797) enthält die Targetsamples (Zielproben), die angeben, welche Zahl zwischen 0 und 9 in dem jeweiligen Graubild dargestellt wird.
    digits = load_digits(return_X_y = True)
    anzahlSamples = 1797

    # - 2.) Extrahieren des Datensets in Input und Target Arrays ------
    # Der erste Eintrag des Tupels enthält die 1797 Bilder und der zweite Eintrag die Targetwerte der Bilder.
    inputs = digits[0]
    targets = digits[1]


    # - 3.) Plotten eines Bildes ----------------------------------
    # Das übergebene Bild aus inputs ander Indexposition 8 zeigt die Zahl 8, welche dem zugehörigen Label in targets an der Indexposition 8 entspricht.
    #plotteEinGraubild(inputs[8])


    # - 4.) Reshapen der Bilder ----------------------------------------
    # Das geforderte Reshapen der Bilder in 8x8 Vektoren der Shape (64) oder (1, 64) ist nicht notwendig, da dies die ursprüngliche Form der Daten ist.


    # - 5.) Reskalierung (Normalisierung) auf den Wertebereich 0 bis 1 als 32 Bit Float Werte ----------------------------------------
    # print(inputs.dtype)  # dtype ist float 64
    inputs = np.float32(inputs)  # von float 64 auf float 32 umwandeln
    # print(inputs.dtype)  # dtype ist float 32
    inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))  # Normalisierung durchführen, wobei 1 = max und 0 = min darstellt


    # - 6.) One-Hot Encoding ----------------------------------------
    # Erzeuge ein Array mit 1797 Zeilen und 10 Spalten, wobei alle Werte mit 0 initialisiert werden
    encodedTargets = np.zeros((anzahlSamples, 10), dtype = np.int8,)  # shape, dtype = float  

    # Setze nach dem Schema der One-Hot Codierung an der entsprechenden Stelle eine 1
    for i in range(anzahlSamples):
        if targets[i] == 0:
            encodedTargets[i][0] = 1
        elif targets[i] == 1:
            encodedTargets[i][1] = 1
        elif targets[i] == 2:
            encodedTargets[i][2] = 1
        elif targets[i] == 3:
            encodedTargets[i][3] = 1
        elif targets[i] == 4:
            encodedTargets[i][4] = 1
        elif targets[i] == 5:
            encodedTargets[i][5] = 1
        elif targets[i] == 6:
            encodedTargets[i][6] = 1
        elif targets[i] == 7:
            encodedTargets[i][7] = 1
        elif targets[i] == 8:
            encodedTargets[i][8] = 1    
        elif targets[i] == 9:
            encodedTargets[i][9] = 1
    

    return (inputs, encodedTargets)




# - 3.) Plotten der Bilder ----------------------------------------
def plotteEinGraubild(inpu: np.ndarray):
    plt.gray()  # Colormap auf Grau setzen
    # Für das Plotten eines Samples müssen die 1D 64 Grauwerte in ein 2D 8x8 Array überführt werden. -> reshape()
    # Das 8x8 Array eines Bildes wird als Matrix in einer neuen Figure dargestellt. -> matshow()
    plt.matshow(inpu.reshape(8, 8))
    plt.show()  # Ohne die leere Figue 1 anzuzeigen, wird auch die neue Figure 2 mit dem übergebenen Graubild nicht angezeigt.



# - 7.) & 8.) Generator Funktion zum Shuffeln & Erzeugen von Minibatches ----------------------------------------
def generatorFunktion(minibatchsize: int, inputsAlt: np.ndarray, targetsAlt: np.ndarray, anzahlSamples: int):
    # Erzeugen einer Liste mit den Zahlen 0 bis 1796
    indexListe = list(range(anzahlSamples))
    # print("IndexListe vor shuffle", indexListe)
    shuffle(indexListe)  # Die shuffle()-Methode nimmt eine Liste und reorganisiert die Reihenfolge der Elemente.
    # print("IndexListe nach shuffle", indexListe)

    # Die durcheinander gewürfelte indexListe wird verwendet, um neue Listen für die Inputs und Targets zu erstellen.
    # Die Reihenfolge der Grauwertbilder in der neuen Input- und Targetliste entspricht der durcheinander gewürfelten Reihenfolge der indexListe.
    # Dadurch, dass sowohl für die Inputs als auch für die Targets die selbe indexListe verwendet wird, bleiben die Input-Targetpaare zusammen an der selben Indexstelle.
    # Die Reihenfolge der Grauwertbilder betrifft jeweils nur die erste Dimension.
    inputsNeu  = inputsAlt[indexListe, :]
    targetsNeu = targetsAlt[indexListe, :]

    # Berechne, wie viele Minibatches basierend auf der Größer der minibatches aus dem Datenset extrahiert werden können.
    anzahlBatches = int(anzahlSamples / minibatchsize)  # abrunden, da für den letzten Minibatch zu wenig Daten vorhanden sind
    start = 0
    stopp = minibatchsize

    # Entsprechend der anzahlBatches viele Minibatches der Größe minibatchsize erstellen und yielden.
    for i in range(anzahlBatches):
        yield (inputsNeu[start:stopp, :], targetsNeu[start:stopp, :])
        start += minibatchsize
        stopp += minibatchsize
