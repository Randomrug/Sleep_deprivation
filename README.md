import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import seed, random
from csv import reader
from math import exp
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import KFold

def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        next(csv_reader)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = {value: i for i, value in enumerate(unique)}
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def dataset_minmax(dataset):
    return [[min(column), max(column)] for column in zip(*dataset)]

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def activate(weights, inputs):
    return sum(weights[i] * inputs[i] for i in range(len(weights) - 1)) + weights[-1]

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = [transfer(activate(neuron['weights'], inputs)) for neuron in layer]
        for neuron, output in zip(layer, new_inputs):
            neuron['output'] = output
        inputs = new_inputs
    return inputs

def transfer_derivative(output):
    return output * (1.0 - output)

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = [(neuron['output'] - expected[j]) if i == len(network) - 1 else 
                  sum(neuron['weights'][j] * neuron['delta'] for neuron in network[i + 1]) 
                  for j, neuron in enumerate(layer)]
        for j, neuron in enumerate(layer):
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1] if i == 0 else [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
    loss_history = []
    mae_history = []
    for epoch in range(n_epoch):
        total_loss = 0
        total_abs_error = 0
        count = 0    
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for _ in range(n_outputs)]
            expected[row[-1]] = 1
            total_loss += -sum(expected[j] * np.log(outputs[j] + 1e-8) for j in range(n_outputs))
            total_abs_error += sum(abs(expected[j] - outputs[j]) for j in range(n_outputs))
            count += 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)            
        loss_history.append(total_loss)
        mae_history.append(total_abs_error / count)            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.5f}, MAE: {mae_history[-1]:.5f}")
    return loss_history, mae_history

def initialize_network(n_inputs, n_hidden, n_outputs):
    return [
        [{'weights': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)],
        [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
    ]

def predict(network, row):
    return forward_propagate(network, row).index(max(forward_propagate(network, row)))

def evaluate_algorithm(dataset, algorithm, n_folds, l_rate, n_epoch, n_hidden):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    scores = []
    all_loss_history = []
    all_mae_history = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train, test = [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]
        predictions, actuals, loss_history, mae_history = algorithm(train, test, l_rate, n_epoch, n_hidden)
        all_loss_history.append(loss_history)
        all_mae_history.append(mae_history)
        accuracy = accuracy_score(actuals, predictions) * 100
        scores.append(accuracy)
        print(f"Fold {fold+1} Accuracy: {accuracy:.2f}%\n\n")    
    avg_loss_history = np.mean(all_loss_history, axis=0)
    avg_mae_history = np.mean(all_mae_history, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_epoch), avg_loss_history, label="Average Loss", color='red')
    plt.plot(range(n_epoch), avg_mae_history, label="Average MAE", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Average Loss and MAE Over Epochs")
    plt.legend()
    plt.show()
    return scores

def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set(row[-1] for row in train))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    loss_history, mae_history = train_network(network, train, l_rate, n_epoch, n_outputs)    
    predictions = [predict(network, row) for row in test]
    actuals = [row[-1] for row in test]
    print("\nSample Predictions:")
    for i in range(10):
        print(f"Actual: {actuals[i]}, Predicted: {predictions[i]}")
    mse = mean_squared_error(actuals, predictions)
    print(f"\nMean Squared Error: {mse:.5f}\n")
    return predictions, actuals, loss_history, mae_history

seed(1)
filename = r"ypur_file_path.csv"
dataset = load_csv(filename)

for i in range(len(dataset[0]) - 1):
    if i == len(dataset[0]) - 5:
        continue
    str_column_to_float(dataset, i)

str_column_to_int(dataset, len(dataset[0]) - 5)

for row in dataset:
    row[-1] = 0 if row[1] < 5 else 1 if row[1] < 15 else 2

normalize_dataset(dataset, dataset_minmax(dataset))

n_folds, l_rate, n_epoch, n_hidden = 5, 0.1, 1000, 10

scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
