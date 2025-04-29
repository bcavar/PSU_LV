from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 784) / 255.0
x_test_s = x_test.reshape(-1, 784) / 255.0
y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

# TODO: strukturiraj konvolucijsku neuronsku mrezu
model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TODO: definiraj callbacks
checkpoint =[ keras.callbacks.TensorBoard(log_dir = 'logs',update_freq = 100),
 keras.callbacks.ModelCheckpoint(filepath='best_model.h5',monitor='val_accuracy',mode='max', save_best_only=True)]
early_stopping = callbacks.EarlyStopping(patience=5, restore_best_weights=True)


# TODO: provedi treniranje mreze pomocu .fit()
history = model.fit(x_train_s, y_train_s,
                    validation_split=0.2,
                    epochs=20,
                    batch_size=128,
                    callbacks=[checkpoint, early_stopping],
                    verbose=2)


#TODO: Ucitaj najbolji model
best_model = keras.models.load_model("best_model.h5")

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_preds = best_model.predict(x_train_s)
test_preds = best_model.predict(x_test_s)

train_acc = accuracy_score(np.argmax(y_train_s, axis=1), np.argmax(train_preds, axis=1))
test_acc = accuracy_score(np.argmax(y_test_s, axis=1), np.argmax(test_preds, axis=1))

print(f"Točnost na trening skupu: {train_acc:.4f}")
print(f"Točnost na test skupu: {test_acc:.4f}")


# TODO: Prikazite matricu zabune na skupu podataka za testiranje
train_cm = confusion_matrix(np.argmax(y_train_s, axis=1), np.argmax(train_preds, axis=1))
test_cm = confusion_matrix(np.argmax(y_test_s, axis=1), np.argmax(test_preds, axis=1))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrica zabune - Trening skup")
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")

plt.subplot(1, 2, 2)
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens')
plt.title("Matrica zabune - Test skup")
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")

plt.tight_layout()
plt.show()
