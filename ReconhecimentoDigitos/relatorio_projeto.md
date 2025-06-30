# Relatório do Projeto: Reconhecimento de Dígitos com Keras

## 1. Introdução
Este projeto tem como objetivo aplicar conceitos de Deep Learning utilizando a biblioteca Keras para treinar um modelo capaz de reconhecer dígitos manuscritos (0 a 9) do conjunto de dados MNIST. O treinamento é realizado em Python 3 e o modelo é salvo no formato `.keras`.

## 2. Funcionamento do Código

### 2.1 `treinamento_mnist.py`
Este é o script principal de treinamento:
- Carrega o dataset MNIST via `keras.datasets`.
- Normaliza os dados de entrada (escala de 0 a 1).
- Realiza a codificação one-hot nos rótulos (labels).
- Cria uma rede neural sequencial com:
  - Camada `Flatten` (entrada 28x28)
  - Duas camadas `Dense` escondidas (128 e 64 neurônios, ReLU)
  - Camada de saída `Dense` com 10 neurônios (softmax)
- Compila e treina o modelo com `categorical_crossentropy` e otimizador Adam
- Avalia a acurácia do modelo com os dados de teste
- Salva o modelo em `modelo_mnist.keras`

### 2.2 `exemplo_previsao.py`
Script que utiliza o modelo salvo para fazer previsões:
- Carrega o modelo `.keras`
- Seleciona aleatoriamente 10 imagens do conjunto de teste
- Exibe cada imagem com a previsão do modelo e o valor real

## 3. Instruções de Execução

### 3.1 Treinamento
Para executar o treinamento:
```bash
python treinamento_mnist.py
```
Isso gerará o arquivo `modelo_mnist.keras` contendo o modelo treinado. Esse passo é opcional para o professor, pois o modelo treinado já é disponibilizado.

### 3.2 Reconhecimento
Para rodar o reconhecimento com exemplos prontos:
```bash
python exemplo_previsao.py
```
Isso exibira 10 imagens com os respectivos dígitos previstos e reais.

## 4. Execução pelo Professor
- Baixe ou clone o repositório com todos os arquivos
- Instale as dependências:
```bash
pip install tensorflow matplotlib
```
- Execute `exemplo_previsao.py`
- O modelo `modelo_mnist.keras` já está pronto e não precisa ser re-treinado




