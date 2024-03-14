import argparse
import re
import matplotlib.pyplot as plt
import os

def extract_data_from_log(log_file_path):
    with open(log_file_path, "r") as file:
        logs = file.readlines()

    epochs = []
    training_accuracies = []
    validation_accuracies = []

    for log in logs:
        # Extract accuracy after every epoch
        match = re.search(r'Epoch \[(\d+)/\d+\] - Training Accuracy: ([\d+.]+)%, Validation Accuracy: ([\d+.]+)%,', log)
        if match:
            epoch = int(match.group(1))
            training_accuracy = float(match.group(2))
            validation_accuracy = float(match.group(3))

            epochs.append(epoch)
            training_accuracies.append(training_accuracy)
            validation_accuracies.append(validation_accuracy)

    return epochs, training_accuracies, validation_accuracies


def save_accuracies(epochs, training_accuracies, validation_accuracies, save_path):
    with open(save_path, "w") as file:
        for epoch, train_acc, val_acc in zip(epochs, training_accuracies, validation_accuracies):
            file.write(f"Epoch: {epoch}, Training Accuracy: {train_acc}, Validation Accuracy: {val_acc}\n")
            print(f"Epoch: {epoch}, Training Accuracy: {train_acc}, Validation Accuracy: {val_acc}")


def visualize_data(epochs, training_accuracies, validation_accuracies, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_accuracies, label="Training Accuracy", color='blue')
    plt.plot(epochs, validation_accuracies, label="Validation Accuracy", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracies over Epochs")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(save_path, format='png')
    plt.close()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize log data")
    parser.add_argument("log_file_path", type=str, help="Path to the log file")
    
    args = parser.parse_args()
    
    epochs, training_accuracies, validation_accuracies = extract_data_from_log(args.log_file_path)
    training_accuracies.insert(0, 0)
    validation_accuracies.insert(0, 0)
    epochs = [e + 1 for e in epochs] 
    epochs.insert(0,0)
    
    # Extract directory and create an output image path
    log_directory = os.path.dirname(args.log_file_path)
            
    accuracies_save_path = os.path.join(log_directory, "accuracies.txt")
    save_accuracies(epochs, training_accuracies, validation_accuracies, accuracies_save_path)

    image_save_path = os.path.join(log_directory, "accuracy_plot.png")
    visualize_data(epochs, training_accuracies, validation_accuracies, image_save_path)
