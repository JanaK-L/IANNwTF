import tensorflow_datasets as tfds
import tensorflow as tf
import tqdm  # wenn wir über etwas iterieren, gibt tqdm uns eine Progressbar
import datetime  # um Zeitstempel beim Loggen mitzuspeichern
import pprint  # pretty print

def load_and_prep_cifar(batch_size, shuffle_buffer_size):
  (train, test), info = tfds.load("cifar10", split = ["train", "test"], as_supervised = True, with_info = True)

  # print("\n\nInformationen zum Datenset:\n", info)
  # tfds.show_examples(train , info)  # Anzeigen einiger Samples mit entsprechendem Label
  # tfds.as_dataframe(train.take(15), info)  # Anzeigen von 15 Samples als Datenframe


  def preprocessing_function(img, label):
    img = tf.cast(img, tf.float32)
    img = (img / 128) - 1  # pseudo normalisation, eigentlich müsste man checken, dass Normalverteilung vorliegt
    label = tf.one_hot(label, depth = 10)
    return img, label


  train = train.map(lambda img, label: preprocessing_function(img, label))
  train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(2)

  test = test.map(lambda img, label: preprocessing_function(img, label))
  test = test.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(2)

  return train, test


# Definition eines Basic CNN Modells mit Logging und Tensorboard
class CIFAR_Modell(tf.keras.Model):

  # Konstruktor Funktion
  def __init__(self, output_size: int = 10):
    # Aufruf des Konstruktors der Superklasse
    super().__init__()  # seit Python 3.0 können die Parameter für super(CIFAR_Modell, self).__init__() weggelassen werden

    # Der Optimizer, die Metric und die Lossfunktion werden als Attribute des Modells gespeichert. Dies ist wichtig für model.compile und model.fit
    self.optimizer = tf.keras.optimizers.Adam()
    #self.optimizer = tf.keras.optimizers.RMSprop()

    # Bei der CategoricalCrossentropy ist from_logits by default = False, welches bedeutet, dass eine erwartet wird, dass y_pred in Form einer Wahrscheinlichkeitsverteilung
    # vorliegt. Dies ist der Fall, wenn wir die Softmax Funktion als Aktivierungsfunktion im Outputlayer verwenden. Wenn wir aber None als Aktivierungsfunktion im Outputlayer
    # benutzen, dann müssen wir from_logits = True setzen, da dann y_pred dann ein logits Tensor ist und keine Wahrscheinlichkeitsverteilung.
    # self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
    self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = False)

    # Liste von Metriken erstellen, dabei ist es egal, wie wir das Attribut nennen, da über self.metrics immer die Metriken gefunden werden
    self.metrics_list = [
      tf.keras.metrics.Mean(name = "loss"),  # Mean um den durchschnittlichen Loss zu berechnen
      tf.keras.metrics.CategoricalAccuracy(name = "accuracy"),  # um die kategorische Gesamtgenauigkeit der Klassifikation zu berechnen
      # Wahrscheinlichkeit, dass das richtige Label in den Top 3 Predictions enthalten ist, in der Regel viel höher als die kategorische Genauigkeit
      tf.keras.metrics.TopKCategoricalAccuracy(3, name = "top-3-accuracy")
    ]

    # Hidden Layer erstellen, das keras.Model wird eine Liste mit Layern erstellen, die über self.layers verfügbar ist, obwohl wir selbst keine Liste erzeugen
    self.convLayer1 = tf.keras.layers.Conv2D(filters = 24, kernel_size = 3, padding = "same", activation = "relu")  # output shape: (batch_size, 32, 32, 24)
    self.convLayer2 = tf.keras.layers.Conv2D(filters = 24, kernel_size = 3, padding = "same", activation = "relu")  # output shape: (batch_size, 32, 32, 24)

    self.pooling = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2)  # halbieren der X und Y Dimensionsgröße der Activationmaps, output shape: (batch_size, 16, 16, 24)
    self.convLayer3 = tf.keras.layers.Conv2D(filters = 48, kernel_size = 3, padding = "same", activation = "relu")  # output shape: (batch_size, 16, 16, 48)
    self.convLayer4 = tf.keras.layers.Conv2D(filters = 48, kernel_size = 3, padding = "same", activation = "relu")  # output shape: (batch_size, 16, 16, 48)

    #self.pooling2 = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2)  # halbieren der X und Y Dimensionsgröße der Activationmaps, output shape: (batch_size, 8, 8, 48)
    #self.convLayer5 = tf.keras.layers.Conv2D(filters = 96, kernel_size = 3, padding = "same", activation = "relu")  # output shape: (batch_size, 8, 8, 96)
    #self.convLayer6 = tf.keras.layers.Conv2D(filters = 96, kernel_size = 3, padding = "same", activation = "relu")  # output shape: (batch_size, 8, 8, 96)

    self.global_pooling = tf.keras.layers.GlobalAvgPool2D()  # 48 Werte in einem Vektor, output ist ein feature_vector der shape: (batch_size, 48)

    # Outputlayer als Dense Layer erzeugen
    self.output_layer = tf.keras.layers.Dense(units = output_size, activation = tf.nn.softmax)  # output shape: (batch_size, 10)


  # Feed Forward Funktion: Berechnung des Hinwegs, wobei das Ergebnis eines Layers an den
  # nachfolgenden Layer weitergegeben wird. Der Übergabeparameter x steht für die Inputwerte.
  @tf.function
  def call(self, x, training = False):
    x = self.convLayer1(x)
    x = self.convLayer2(x)
    x = self.pooling(x)
    x = self.convLayer3(x)
    x = self.convLayer4(x)
    #x = self.pooling2(x)
    #x = self.convLayer5(x)
    #x = self.convLayer6(x)
    x = self.global_pooling(x)
    x = self.output_layer(x)
    return x


  # Funktion, um die Metrics zu reseten
  def reset_metrics(self):
    for metric in self.metrics:
      metric.reset_states()


  # Funktion zum Trainieren, jeder train_step Aufruf erhält 1x Batch of data
  @tf.function
  def train_step(self, data):
    x, targets = data  # data ist ein Tupel aus Inputs x und den Targets

    # GradientTape öffnen und Forward Step und Loss berechnen
    with tf.GradientTape() as tape:
      predictions = self(x, training = True)   # Aufruf der call Funktion (Forward Step)
      loss = self.loss_function(targets, predictions)  # Berechnung des Losses

    # Gradienten mit Respekt auf die trainable_variables und den Loss berechnen
    gradients = tape.gradient(loss, self.trainable_variables)

    # Anwenden der Gradienten = Updaten der Gewichte (trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # updaten der Metrics
    self.metrics[0].update_state(loss)  # updaten der Loss Metrik
    for metric in self.metrics[1:]:  # updaten aller Metriken außer der Loss Metrik
      metric.update_state(targets, predictions)

    # zurückgeben eines Dictionarys, welches die Metric Namen und das entsprechende aktuelle Result zurückgibt
    # mit Hilfe einer List Comprehension wird über die Metrics Liste iteriert und der Name der Metrik wird als Key und der Wert als Value verwendet
    return {m.name: m.result() for m in self.metrics}


  # Funktion zum Testen
  @tf.function
  def test_step(self, data):
    x, targets = data
    predictions = self(x, training = False)
    loss = self.loss_function(targets, predictions)

    # updaten der Metrics (genau wie im Training Step)
    self.metrics[0].update_state(loss)  # updaten der Loss Metrik
    for metric in self.metrics[1:]:  # updaten aller Metriken außer der Loss Metrik
      metric.update_state(targets, predictions)

    return {m.name: m.result() for m in self.metrics}



def training_loop(num_epochs, batch_size, shuffle_buffer_size, lr, train_summary_writer, test_summary_writer):
    model = CIFAR_Modell(output_size = 10)  # Modell erstellen
    train, test = load_and_prep_cifar(batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size)  # Starte Preprocessing des Trainings- und des Testdatensets

    # model(tf.keras.Input((32, 32, 1)))  # Dummy Input in das Modell werfen, damit die Layer gebaut werden und man die Summary des Modells angucken kann
    # model.summary()  # die non trainable Parameters in der Summary müssten die 6 Metriken sein!!!


    for epoch in range(num_epochs):
        print(f"Epoch {epoch}:")

        # Training:
        for data in tqdm.tqdm(train, position=0, leave=True):
            metrics = model.train_step(data)  # der train_step gibt ein Dictionary mit den Metriken und den entsprechenden Werten zurück, aber wir benutzen sie nicht, TODO?!

            # loggen/speichern der Train Metrik einer Batch in der log Datei, welche von Tensorboard benutzt wird
            with train_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step = epoch)

        # printen aller Metriken
        print([f"Train_{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # Reseten aller Metriken (benötigt eine reset_metrics Methode in dem Modell)
        model.reset_metrics()



        # Testen bzw. Validierung:
        for data in test:
            metrics = model.test_step(data)

            # loggen/speichern der Test Metrik einer Batch in der log Datei, welche von Tensorboard benutzt wird
            with test_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        # printen aller Metriken, jedoch dieses mal mit dem Präfix Test_
        print([f"Test_{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # Reseten aller Metriken
        model.reset_metrics()


        print("\n")

    return model    



# In einer Konfigurationsdatei sollten alle Informationen über die verwendete Architektur und die Hyperparameter gespeichert werden.
# Der Name der Konfigurationsdatei wird als Subfolder verwendet.
config_name= "Run-01"

# akutellen Zeitpunkt im Format Jahr, Monat, Tag - Stunden, Minuten, Sekunden
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Tensorboard-Protokolle (Trainings- und Testlogs) werden in einem Ordner mit einem aussagekräftigen Namen (z.B. logs/Name des Trainingslaufs/Datum und Uhrzeit/...) gespeichert
train_log_path = f"logs/{config_name}/{current_time}/train"
test_log_path = f"logs/{config_name}/{current_time}/test"

# Summary Writer erhält als Argument wird den Speicherort
train_summary_writer = tf.summary.create_file_writer(train_log_path)  # Log Writer für Training Metrics
test_summary_writer = tf.summary.create_file_writer(test_log_path)  # Log Writer für Test Metrics


# Einige der Hyperparameter definieren
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 0.00001

# Training starten
model = training_loop(num_epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, shuffle_buffer_size = SHUFFLE_BUFFER_SIZE, lr = LR, train_summary_writer = train_summary_writer, test_summary_writer = test_summary_writer)

