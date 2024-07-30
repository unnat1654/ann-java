# Artificial Neural Network in Java

This project implements an artificial neural network (ANN) in Java, designed for classification tasks using the sigmoid activation function. The network is multithreaded and employs stochastic gradient descent (SGD) to adjust the weights of the neurons. It allows saving and loading trained models, and supports further training (transfer learning).

## Project Overview

The purpose of this project is to enhance understanding of object-oriented programming in Java and to implement a neural network after studying the theoretical aspects.

## Features

- **Sigmoid Activation Function:** Used for classification predictions.
- **Multithreaded Processing:** To improve performance during training.
- **Model Saving and Loading:** Save and load trained models for reuse and transfer learning.
- **Stochastic Gradient Descent:** Optimizes neuron weights during training.

## Getting Started

### Prerequisites

Ensure you have the Java Development Kit (JDK) installed. You can check this by running the following command:

```java
javac -version
```

### Installation

1. **Download the Project**

   Clone or download the project folder into your desired directory.

2. **Compile the Code**

   Navigate to the project directory and compile the Java files:

   ```java
   javac ArtificialNNModel/ArtificialNeuralNetwork.java, ArtificialNNModel/BiMap.java, ArtificialNNModel/CSVReader.java, ArtificialNNModel/inputLayer.java, ArtificialNNModel/Layer.java, ArtificialNNModel/outputLayer.java, ArtificialNNModel/Neuron.java
   ```
3. **Import the Package**

   Import the classes from the package in your Java code:

   ```java
   import ArtificialNNModel.*;
   ```

### Usage

1. **Read Data from CSV**

   Use `CSVReader.readCSV(String filePath)` to read your CSV file into a `List<String[]>`.

   ```java
   List<String[]> data = CSVReader.readCSV("path/to/your/file.csv");
   ```

2. **Prepare Data**

   Convert the read CSV data into a `double[][] inputs` array and a `double[] outputs` array. Ensure that each row in the `inputs` array represents an observation, and each element in the row represents a feature.

3. **Initialize the Neural Network**

   Create an instance of the neural network:

   ```java
   ArtificialNeuralNetwork ann = new ArtificialNeuralNetwork();
   ```

   Set up the network layers:

   - **Add Input Layer:**

     ```java
     ann.addInputLayer(int no_of_classes);
     ```

   - **Add Hidden Layer:**

     ```java
     ann.addHiddenLayer(int number_of_neurons, double learningRate);
     ```

   - **Add Output Layer:**

     ```java
     ann.addOutputLayer(double[] possibleOutputs, double learningRate);
     ```

4. **Train the Model**

   Fit the model to your data:

   ```java
   ann.fit(double[][] inputs, double[] outputs, int epochs);
   ```

5. **Make Predictions**

   Use the trained model to make predictions:

   ```java
   double prediction = ann.predict(double[] input);
   ```

6. **Evaluate Accuracy**

   Calculate the accuracy of the model:

   ```java
   double accuracy = ann.accuracy(double[][] inputs, double[] outputs);
   ```

7. **Save and Load Models**

   - **Save the Model:**

     ```java
     ann.saveWeights(String fileName);
     ```

   - **Load a Saved Model:**

     ```java
     ArtificialNeuralNetwork ann = new ArtificialNeuralNetwork(String filename);
     ```

## Example

Here's a simple example of how to use the neural network:

```java
import ArtificialNNModel.*;

public class Main {
    public static void main(String[] args) {
        // Step 1: Read data
        List<String[]> data = CSVReader.readCSV("data.csv");
        
        // Step 2: Prepare data
        double[][] inputs = ...; // Convert data to inputs array
        double[] outputs = ...;  // Convert data to outputs array

        // Step 3: Initialize and configure the neural network
        ArtificialNeuralNetwork ann = new ArtificialNeuralNetwork();
        ann.addInputLayer(4);  // Example: 4 features
        ann.addHiddenLayer(10, 0.01); // Example: 10 neurons, 0.01 learning rate
        ann.addOutputLayer(new double[]{0, 1}, 0.01); // Example: binary classification

        // Step 4: Train the model
        ann.fit(inputs, outputs, 1000); // Example: 1000 epochs

        // Step 5: Make predictions
        double prediction = ann.predict(new double[]{...});

        // Step 6: Calculate accuracy
        double accuracy = ann.accuracy(inputs, outputs);
        System.out.println("Model Accuracy: " + accuracy);

        // Step 7: Save the model
        ann.saveWeights("model_weights.txt");
    }
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

