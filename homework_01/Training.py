import DataPreprocessing as dp
import MLP
import numpy as np
import AktivierungsUndLossFunktionen as funks
import matplotlib.pyplot as plt


def main():
    # Preprocessing der Daten für alle 1797 Samples durchführen
    inputs, targets = dp.preProcessing()
    anzahlSamples = 1797

    # Subset aus drei Bildern zum testen erstellen 
    # inputs = inputs[0:anzahlSamples, :]
    # print("SubsetInputs", subsetIn)
    # targets = targets[0:anzahlSamples]
    # print("SubsetTargets", subsetTar)

    # erzeuge ein MLP
    minibatchsize = 128
    sizeOfEachLayer = np.array([5, 3, 10])
    lernrate = 0.1
    mlp = MLP.MLP(sizeOfEachLayer, minibatchsize, lernrate)
    cce = funks.CCE()
    anzahlEpochs = 100
    alleLosses = []

    # Iteriere über anzahlEpochs oft über das gesamte Datenset
    for j in range(anzahlEpochs):
        averageLossPerEpoch = []

        # durchlaufe alle Elemente, die durch die Generatorfunktion geyielded werden
        # beachte, dass die Generatorfunktion nur einmal ausgeführt wird 
        for i in dp.generatorFunktion(minibatchsize, inputs, targets, anzahlSamples):
            # Inputs und Targets aus dem Tupel extrahieren
            inputs_batched = i[0]
            targets_batched = i[1]

            # Hinweg berechnen
            klassifikationsErgebnisDesMLP = mlp.hinweg(inputs_batched)
            #print("Ergebnis des MLP Hinwegs:\n", klassifikationsErgebnisDesMLP)

            # Loss berechnen
            loss = cce.call(klassifikationsErgebnisDesMLP, targets_batched)
            #print("Loss:\n", loss, "\n")
            averageLoss = np.average(loss)
            averageLossPerEpoch.append(averageLoss)
            #print("\n", averageLoss, "\n")

            # Rückweg berechnen
            # print("\n", cce.backwards(klassifikationsErgebnisDesMLP, targets))
            mlp.rückweg(klassifikationsErgebnisDesMLP, targets_batched)

            # Updaten der Gewichtsmatrizen und der Biase
            mlp.update()

            #print("\n-------------------------------------------------\n")

        # Genauigkeit berechnen: Achtung nur für den letzten Minibatch des aktuellen Epochs
        # argmax gibt den Index des höchsten Wertes, also den Index der größten Klassifikationswahrscheinlichkeit aus der (minibatchsize, 10) großen ErgebnisMatrix zurück
        # axis = 1, da wir aus jeder Zeile, also für jedes Bild, den Index der größten Klassifikationswahrscheinlichkeit haben wollen
        # wenn der Index der größten Klassifikationswahrscheinlichkeit gleich dem Index des Werte 1 des one Hot encodeten targetvektors entspricht, wurde korrekt klassifiziert = >True
        # über == erhalten wir eine boolean Matrix, welche 0 bei allen falsch klassifizierten Samples stehen hat und eine 1 bei allen richtig klassifizierten
        # über die Summe der boolean Matrix erhalten wir die Anzahl der korrekten Klassifikationen, welche durch die Anzahl an klassifizierten Samples des letzten Minibatches,
        # also die minibatchsize, geteilt wird
        print("ACCURACY des letzten Minibatches von Epoch", j, np.sum(np.argmax(klassifikationsErgebnisDesMLP, axis=1) == np.argmax(targets_batched, axis=1)) / minibatchsize)


        # averageLossPerEpoch in der alleLosses Liste speichern
        averageLossPerEpoch = np.average(np.array(averageLossPerEpoch))
        alleLosses.append(averageLossPerEpoch)


    # Plot the average loss = y vs. the epoch number = x
    plt.plot(range(0, len(alleLosses)), alleLosses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Average Loss per Epoch")
    plt.savefig("AverageLoss")  # speichern des Bildes im aktuellen Ordner
    plt.show()
    


        






if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt => Abbruch")

