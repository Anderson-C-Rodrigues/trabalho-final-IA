import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Carregar o dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalizar os dados
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# 3. One-hot encoding das labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. Criar o modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 5. Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Treinar o modelo
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# 7. Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia no teste: {test_acc:.4f}")

# 8. Salvar o modelo no formato .keras
model.save("modelo_mnist.keras")
print("Modelo salvo como 'modelo_mnist.keras'")

# 9. (Opcional) Visualização da acurácia por época
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('precisão durante o treinamento')
plt.xlabel('Época')
plt.ylabel('precisão')
plt.legend()
plt.grid(True)
plt.show()
