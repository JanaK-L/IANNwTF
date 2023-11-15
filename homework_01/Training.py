import DataPreprocessing as dp
import MLP
import numpy as np
import AktivierungsUndLossFunktionen as funks


def main():
    # Preprocessing der Daten für alle 1797 Samples durchführen
    inputs, targets = dp.preProcessing()

    anzahlSamples = 5  # TODO Subsets weg oder auf 1797 setzen
    # Subset aus drei Bildern zum testen erstellen 
    inputs = inputs[0:anzahlSamples, :]
    # print("SubsetInputs", subsetIn)
    targets = targets[0:anzahlSamples]
    # print("SubsetTargets", subsetTar)

    # erzeuge ein MLP
    minibatchsize = 4
    sizeOfEachLayer = np.array([5, 3, 10])
    mlp = MLP.MLP(sizeOfEachLayer, minibatchsize)
    cce = funks.CCE()

    # durchlaufe alle Elemente, die durch die Generatorfunktion geyielded werden
    # beachte, dass die Generatorfunktion nur einmal ausgeführt wird 
    for i in dp.generatorFunktion(minibatchsize, inputs, targets, anzahlSamples):
        #print("Inputs:\n", i[0])
        #print("Targets:\n", i[1])

        # Inputs und Targets aus dem Tupel extrahieren
        inputs = i[0]
        targets = i[1]

        # Hinweg berechnen
        klassifikationsErgebnisDesMLP = mlp.hinweg(inputs)
        #print("Ergebnis des MLP Hinwegs:\n", klassifikationsErgebnisDesMLP)

        # Loss berechnen
        loss = cce.call(klassifikationsErgebnisDesMLP, targets)
        print("Loss:\n", loss)

        # Rückweg berechnen
        # print("\n", cce.backwards(klassifikationsErgebnisDesMLP, targets))
        mlp.rückweg(klassifikationsErgebnisDesMLP, targets)

        print("\n-------------------------------------------------\n")






if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")




    