import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import os

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Classe para a Rede Neural
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Pesos aleatórios ajustados para evitar saturação
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))

        # Escolha da função de ativação
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative

    def forward(self, X):
        self.input_layer = X
        self.hidden_layer_input = np.dot(self.input_layer, self.weights_input_hidden)
        self.hidden_layer_output = self.activation(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output_layer_output = self.activation(self.output_layer_input)

        return self.output_layer_output

    def backward(self, X, y, output, learning_rate):
        # Cálculo do erro da camada de saída
        output_error = y - output
        output_delta = output_error * self.activation_derivative(output)

        # Cálculo do erro da camada oculta
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_layer_output)

        # Atualização dos pesos
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * learning_rate
        self.weights_input_hidden += np.dot(self.input_layer.T, hidden_delta) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        # Normalização dos dados de entrada para evitar saturação
        X = (X - np.min(X)) / (np.max(X) - np.min(X)) * 2 - 1

        history = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

            # Armazena o erro por iteração
            loss = np.mean(np.square(y - output))
            history.append({"epoch": epoch + 1, "error": loss})

            if epoch % 100 == 0:  # Exibir o erro a cada 100 épocas
                print(f"Época {epoch}, Erro: {loss}")
        return history

    def predict(self, X):
        # Normalização dos dados de entrada
        X = (X - np.min(X)) / (np.max(X) - np.min(X)) * 2 - 1
        return self.forward(X)

# Carregamento de dados
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = pd.get_dummies(data.iloc[:, -1]).values  # Codificar classes em formato binário
    return X, y

# Funções de seleção de arquivo
def select_train_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        train_file_entry.delete(0, tk.END)
        train_file_entry.insert(0, file_path)

def select_test_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        test_file_entry.delete(0, tk.END)
        test_file_entry.insert(0, file_path)

def select_output_directory():
    directory = filedialog.askdirectory()
    if directory:
        output_dir_entry.delete(0, tk.END)
        output_dir_entry.insert(0, directory)

# Função de interface principal
def start_training():
    train_file = train_file_entry.get()
    test_file = test_file_entry.get()
    hidden_neurons = int(hidden_neurons_entry.get())
    activation_choice = activation_var.get()
    epochs = int(epochs_entry.get())
    learning_rate = float(learning_rate_entry.get())
    output_directory = output_dir_entry.get()

    if not output_directory:
        messagebox.showerror("Erro", "Por favor, selecione um diretório para salvar os resultados.")
        return

    activation_func = 'sigmoid' if activation_choice == "Logística" else 'tanh'

    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    nn = NeuralNetwork(input_size, hidden_neurons, output_size, activation=activation_func)

    print("Iniciando treinamento...")
    history = nn.train(X_train, y_train, epochs, learning_rate)

    print("Realizando predições...")
    predictions = nn.predict(X_test)
    predictions_classes = np.argmax(predictions, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    conf_matrix = pd.crosstab(y_test_classes, predictions_classes, rownames=['Classe Real'], colnames=['Classe Prevista'])

    # Salvando a matriz de confusão em um arquivo CSV
    conf_matrix_file = os.path.join(output_directory, "resultado_matriz_confusao.csv")
    conf_matrix.to_csv(conf_matrix_file)

    # Salvando o histórico de iterações e erros
    history_file = os.path.join(output_directory, "resultado_iteracoes.csv")
    pd.DataFrame(history).to_csv(history_file, index=False)

    messagebox.showinfo(
        "Treinamento Concluído",
        f"Treinamento concluído!\nOs resultados foram salvos em:\n{output_directory}"
    )

# Configurando a Interface Gráfica
root = tk.Tk()
root.title("Configuração da Rede Neural Backpropagation")

# Entrada de Arquivos
tk.Label(root, text="Arquivo de Treinamento:").grid(row=0, column=0, padx=10, pady=5)
train_file_entry = tk.Entry(root, width=50)
train_file_entry.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Procurar", command=select_train_file).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Arquivo de Teste:").grid(row=1, column=0, padx=10, pady=5)
test_file_entry = tk.Entry(root, width=50)
test_file_entry.grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Procurar", command=select_test_file).grid(row=1, column=2, padx=10, pady=5)

# Entrada de Parâmetros
tk.Label(root, text="Número de Neurônios na Camada Oculta:").grid(row=2, column=0, padx=10, pady=5)
hidden_neurons_entry = tk.Entry(root, width=10)
hidden_neurons_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Função de Ativação:").grid(row=3, column=0, padx=10, pady=5)
activation_var = tk.StringVar(value="Logística")
tk.Radiobutton(root, text="Logística", variable=activation_var, value="Logística").grid(row=3, column=1, sticky="W")
tk.Radiobutton(root, text="Tangente Hiperbólica", variable=activation_var, value="Tangente Hiperbólica").grid(row=3, column=2, sticky="W")

tk.Label(root, text="Número de Iterações (Épocas):").grid(row=4, column=0, padx=10, pady=5)
epochs_entry = tk.Entry(root, width=10)
epochs_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Taxa de Aprendizado:").grid(row=5, column=0, padx=10, pady=5)
learning_rate_entry = tk.Entry(root, width=10)
learning_rate_entry.grid(row=5, column=1, padx=10, pady=5)

# Diretório de Saída
tk.Label(root, text="Diretório de Saída:").grid(row=6, column=0, padx=10, pady=5)
output_dir_entry = tk.Entry(root, width=50)
output_dir_entry.grid(row=6, column=1, padx=10, pady=5)
tk.Button(root, text="Procurar", command=select_output_directory).grid(row=6, column=2, padx=10, pady=5)

# Botão para Iniciar o Treinamento
train_button = tk.Button(root, text="Iniciar Treinamento", command=start_training)
train_button.grid(row=7, column=0, columnspan=3, pady=20)

root.mainloop()

