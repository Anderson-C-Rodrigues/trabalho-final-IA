import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist


(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255


modelo = load_model("modelo_mnist.keras")


num_imagens = 10
indices = np.random.choice(len(x_test), num_imagens, replace=False)


plt.figure(figsize=(15, 4))

for i, idx in enumerate(indices):
    imagem = x_test[idx]
    imagem_exp = np.expand_dims(imagem, axis=0)
    predicao = modelo.predict(imagem_exp, verbose=0)
    digito_previsto = np.argmax(predicao)

    
    plt.subplot(1, num_imagens, i + 1)
    plt.imshow(imagem, cmap='gray')
    plt.title(f"P: {digito_previsto}\nR: {y_test[idx]}")
    plt.axis('off')

plt.suptitle("Previsões do Modelo - P: previsto | R: real")
plt.tight_layout()
plt.show()

