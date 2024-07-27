package ArtificialNNModel;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.lang.IllegalArgumentException;

enum Caller {
    USER,
    MODEL
}

public class ArtificialNeuralNetwork {
    Layer levels[];
    public double outputOfNeurons[][];
    public int outputNeuronsCount = 0;
    public int inputClassesCount = 0;
    private String type;
    private HashMap<Double, Integer> map = new HashMap<>();

    private void backPropogation() {
        System.out.println("Starting backpropagation.");
        this.levels[levels.length - 1].receiveFrontError(this.outputOfNeurons);
        for (int i = levels.length - 1; i >= 2; i--) { // lower bound is 2 since input layer does not need to adjust
                                                       // weights
            this.levels[i - 1].receiveFrontError(levels[i].sendBackError());
        }
        System.out.println("Backpropagation finished.");
    }

    private int arrayMaxIndex(double[] array) {
        double max = 0;
        int index = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }
        System.out.println("Max index found: " + index);
        return index;
    }

    public void addInputLayer(int no_of_classes) throws IllegalArgumentException {
        System.out.println("Adding input layer with " + no_of_classes + " neurons.");
        if (no_of_classes <= 0) {
            throw new IllegalArgumentException("Number of input neurons must be greater than zero.");
        }
        this.levels = new Layer[1];
        this.levels[this.levels.length - 1] = new inputLayer(no_of_classes);
        this.inputClassesCount = no_of_classes;
    }

    public void addHiddenLayer(int no_of_neurons, double learningRate)
            throws IllegalStateException, IllegalArgumentException {
        System.out.println("Adding hidden layer with " + no_of_neurons + " neurons and learning rate " + learningRate);
        if (this.inputClassesCount == 0) {
            throw new IllegalStateException("Input Layer must be initialized before adding a hidden layer");
        }
        if (no_of_neurons <= 0) {
            throw new IllegalArgumentException("Number of neurons must be greater than zero.");
        }
        Layer[] temp = this.levels;
        this.levels = new Layer[temp.length + 1];
        for (int i = 0; i < temp.length; i++) {
            this.levels[i] = temp[i];
        }
        int prevLayerNoNeurons = this.levels[temp.length - 1].numberOfNeurons;
        this.levels[this.levels.length - 1] = new Layer(no_of_neurons, learningRate, prevLayerNoNeurons);
    }

    public void addOutputLayer(String type, int outputNeuronsCount, double learningRate)
            throws IllegalArgumentException, IllegalStateException {
        System.out.println("Adding output layer with type " + type + ", " + outputNeuronsCount
                + " neurons, and learning rate " + learningRate);
        if (this.inputClassesCount == 0) {
            throw new IllegalStateException("Input Layer must be initialized before adding a hidden layer");
        }
        if (!"regression".equals(type) && !"classification".equals(type)) {
            throw new IllegalArgumentException("The 'type' property can be 'classification' or 'regression' only.");
        }
        if (outputNeuronsCount <= 0) {
            throw new IllegalArgumentException("Number of output neurons must be greater than zero.");
        }

        Layer[] temp = this.levels;
        this.type = type;
        this.levels = new Layer[temp.length + 1];
        for (int i = 0; i < temp.length; i++) {
            this.levels[i] = temp[i];
        }

        int prevLayerNoNeurons = this.levels[temp.length - 1].numberOfNeurons;
        if ("regression".equals(this.type)) {
            this.levels[this.levels.length - 1] = new Layer(1, learningRate, prevLayerNoNeurons);
            this.outputNeuronsCount = 1;
            return;
        }
        this.outputNeuronsCount = Math.max(outputNeuronsCount, 1);
        this.levels[this.levels.length - 1] = new Layer(this.outputNeuronsCount, learningRate, prevLayerNoNeurons);
    }

    public double[] predict(double input[]) throws IllegalArgumentException, IllegalStateException {
        System.out.println("Predicting for input: " + Arrays.toString(input));
        if (outputNeuronsCount == 0) {
            throw new IllegalStateException(
                    "The neural network layers not set up properly. Input and output layers are required.");
        }
        if (input.length != inputClassesCount) {
            throw new IllegalArgumentException(
                    "The numbers of classes in input layer do not match classes in entered array.");
        }
        levels[0].calculateOutputs(input,Caller.USER);
        for (int i = 1; i < levels.length; i++) {
            levels[i].calculateOutputs(levels[i - 1].neuronOutputs, Caller.USER);
        }
        System.out.println("Prediction result: " + Arrays.toString(levels[levels.length - 1].neuronOutputs));
        return levels[levels.length - 1].neuronOutputs;
    }

    private double[] comp_predict(double input[]) throws IllegalArgumentException, IllegalStateException {
        System.out.println("Predicting for input: " + Arrays.toString(input));
        if (outputNeuronsCount == 0) {
            throw new IllegalStateException(
                    "The neural network layers not set up properly. Input and output layers are required.");
        }
        if (input.length != inputClassesCount) {
            throw new IllegalArgumentException(
                    "The numbers of classes in input layer do not match classes in entered array.");
        }
        levels[0].calculateOutputs(input, Caller.MODEL);
        for (int i = 1; i < levels.length; i++) {
            levels[i].calculateOutputs(levels[i - 1].neuronOutputs, Caller.MODEL);
        }
        System.out.println("Prediction result: " + Arrays.toString(levels[levels.length - 1].neuronOutputs));
        return levels[levels.length - 1].neuronOutputs;
    }

    private double forwardPropogation_accuracy(double inputs[][], double outputs[]) throws IllegalArgumentException, IllegalStateException {
        System.out.println("Calculating accuracy...");
        if (this.outputNeuronsCount == 0) {
            throw new IllegalStateException(
                    "The neural network layers not set up properly. Input and output layers are required.");
        }
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("Number of observations and labels do not match");
        }
        int correctAnswers = 0;
        if ("classification".equals(this.type)) {
            for (int i = 0; i < inputs.length; i++) {
                int predictionIndex = arrayMaxIndex(this.comp_predict(inputs[i]));
                if (map.get(outputs[i]) == null) {
                    System.out.println(outputs[i]);
                    throw new IllegalArgumentException("Output does not match labels");
                }
                if (predictionIndex == map.get(outputs[i]))
                    correctAnswers++;
            }
            double accuracy = (double) correctAnswers / outputs.length;
            System.out.println("Classification accuracy: " + accuracy);
            return accuracy;
        }
        double error = 0;
        for (int i = 0; i < inputs.length; i++) {
            double prediction = this.comp_predict(inputs[i])[0];
            error += Math.pow(outputs[i] - prediction, 2);
        }
        double mse = error / outputs.length;
        System.out.println("Regression Mean Squared Error: " + mse);
        return mse;
    }


    public double accuracy(double inputs[][], double outputs[]) throws IllegalArgumentException, IllegalStateException {
        System.out.println("Calculating accuracy...");
        if (this.outputNeuronsCount == 0) {
            throw new IllegalStateException(
                    "The neural network layers not set up properly. Input and output layers are required.");
        }
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("Number of observations and labels do not match");
        }
        int correctAnswers = 0;
        if ("classification".equals(this.type)) {
            for (int i = 0; i < inputs.length; i++) {
                int predictionIndex = arrayMaxIndex(this.predict(inputs[i]));
                if (map.get(outputs[i]) == null) {
                    System.out.println(outputs[i]);
                    throw new IllegalArgumentException("Output does not match labels");
                }
                if (predictionIndex == map.get(outputs[i]))
                    correctAnswers++;
            }
            double accuracy = (double) correctAnswers / outputs.length;
            System.out.println("Classification accuracy: " + accuracy);
            return accuracy;
        }
        double error = 0;
        for (int i = 0; i < inputs.length; i++) {
            double prediction = this.predict(inputs[i])[0];
            error += Math.pow(outputs[i] - prediction, 2);
        }
        double mse = error / outputs.length;
        System.out.println("Regression Mean Squared Error: " + mse);
        return mse;
    }

    public void fit(double inputs[][], double outputs[], int epochs) throws IllegalArgumentException {
        System.out.println("Running for " + outputs.length + " observations.");
        System.out.println("Fitting model with " + epochs + " epochs.");
        if (this.outputNeuronsCount == 0) {
            throw new IllegalStateException(
                    "The neural network layers not set up properly. Input and output layers are required.");
        }
        if (inputs[0].length != levels[0].numberOfNeurons || inputs.length != outputs.length) {
            throw new IllegalArgumentException("No of classes in input are not valid");
        }
        this.outputOfNeurons = new double[this.outputNeuronsCount][outputs.length];
        if ("regression".equals(this.type)) {
            this.outputOfNeurons[0] = outputs;
        }
        if ("classification".equals(this.type)) {
            int i = 0;
            for (double output : outputs) {
                if (map.size() == this.outputNeuronsCount)
                    break;
                if (!map.containsKey(output)) {
                    if (map.size() < this.outputNeuronsCount)
                        map.put(output, i++);
                    else
                        throw new IllegalArgumentException(
                                "The output array contains more unique outputs than provided in NoOutputNeurons in output Layer initiailization.");
                }
            }
            for (i = 0; i < outputs.length; i++) {
                this.outputOfNeurons[map.get(outputs[i])][i] = 1;
            }
            map.forEach((k, v) -> {
                System.out.println(k + " -> " + v);
            });
        }
        while (0 < epochs--) {
            System.out.print("Epochs left: " + epochs);
            double accuracy = forwardPropogation_accuracy(inputs, outputs);
            System.out.println("Model Accuracy: " + accuracy);
            backPropogation();
        }
        System.out.println("Model fitting complete.");
    }
}

class Layer {
    protected Neuron neurons[];
    private int prevLayerNeuronsCount = 1;

    private void generateNeurons() {
        System.out.println("Generating neurons for layer with " + this.numberOfNeurons + " neurons.");
        for (int i = 0; i < this.numberOfNeurons; i++) {
            this.neurons[i] = new Neuron(this.learningRate, this.Method, this.prevLayerNeuronsCount);
            System.out.println("Generated neuron " + (i + 1) + " with method " + this.Method + " and learning rate "
                    + this.learningRate);
        }
    }

    public String Method = "Relu"; // default
    public double learningRate = 0.01; // default
    public int numberOfNeurons = 1; // default
    public double neuronOutputs[];

    // Constructors
    public Layer(int number, double learningRate, int prevLayerNoNeurons) {
        System.out.println("Creating layer with " + number + " neurons, learning rate " + learningRate
                + ", previous layer neuron count " + prevLayerNoNeurons);
        this.neurons = new Neuron[number];
        this.numberOfNeurons = number;
        this.learningRate = learningRate;
        this.prevLayerNeuronsCount = prevLayerNoNeurons;
        this.neuronOutputs = new double[number];
        generateNeurons();
    }

    public Layer(int number, double learningRate, String Method, int prevLayerNoNeurons) {
        System.out.println("Creating layer with " + number + " neurons, learning rate " + learningRate + ", method "
                + Method + ", previous layer neuron count " + prevLayerNoNeurons);
        this.neurons = new Neuron[number];
        this.numberOfNeurons = number;
        this.learningRate = learningRate;
        this.Method = Method;
        this.neuronOutputs = new double[number];
        this.prevLayerNeuronsCount = prevLayerNoNeurons;
        generateNeurons();
    }

    public void calculateOutputs(double inputs[], Caller caller) { // inputs=[cl1,cl2,cl3,cl4,cl5...]

        System.out.println("Calculating outputs for inputs: " + Arrays.toString(inputs));
        for (int i = 0; i < this.numberOfNeurons; i++) {
            this.neuronOutputs[i] = this.neurons[i].output(inputs, caller);
            System.out.println("Output for neuron " + (i + 1) + ": " + this.neuronOutputs[i]);
        }
    }

    public void receiveFrontError(double error[][]) { // error=[thisLayerneuronnumber][DataItemsNo] or
                                                      // [[n1o1,n1o2,n1o3,n1o4,n1o5...],[n2o1,n2o2,n2o3...]..]
        System.out.println("Receiving front error: " + Arrays.deepToString(error));
        for (int i = 0; i < this.numberOfNeurons; i++) {
            this.neurons[i].gradientDescent(error[i]);
            System.out.println("Performed gradient descent for neuron " + (i + 1));
        }
    }

    public double[][] sendBackError() {
        System.out.println("Sending back error to the previous layer.");
        int numberOfObservations = this.neurons[0].errorForPreviousLayer.length;
        int prevLayerNeuronsCount = this.neurons[0].errorForPreviousLayer[0].length;

        double[][][] thisLayerError = new double[numberOfNeurons][numberOfObservations][prevLayerNeuronsCount];
        for (int i = 0; i < this.neurons.length; i++) {
            thisLayerError[i] = this.neurons[i].errorForPreviousLayer;
        }

        double[][] addedThisLayerError = new double[prevLayerNeuronsCount][numberOfObservations];
        for (int i = 0; i < prevLayerNeuronsCount; i++) {
            for (int j = 0; j < numberOfObservations; j++) {
                for (int k = 0; k < numberOfNeurons; k++) {
                    addedThisLayerError[i][j] += this.neurons[k].errorForPreviousLayer[j][i];
                }
            }
        }
        System.out.println("Back error calculated: " + Arrays.deepToString(addedThisLayerError));
        return addedThisLayerError;
    }
}

class inputLayer extends Layer {
    public inputLayer(int number) {
        super(number, 0, "input", 1);
    }

    @Override
    public void calculateOutputs(double inputs[], Caller caller) { // the input is [cl1,cl2,cl3,cl4,cl5....]
        System.out.println("Calculating outputs for input layer with inputs: " + Arrays.toString(inputs));
        for (int i = 0; i < super.numberOfNeurons; i++) {
            double[] input = { inputs[i] }; // input=[cl1]
            super.neuronOutputs[i] = super.neurons[i].output(input, caller);
            System.out.println("Input layer output for neuron " + (i + 1) + ": " + super.neuronOutputs[i]);
        }
    }
}

class Neuron {
    private double weights[];
    private double bias = 0;
    private double learningRate;
    private double previousInputs[][]; // previousInputs[DataItemsNo][PrevoiusLayerNeuronNumber]
    public double errorForPreviousLayer[][]; // errorForBack=[[ob1weighterror1,ob1we2,ob1we3...],[ob2we1,ob2we2...]...]
    public String Method = "Relu"; // enum: "input" and "Relu"

    // returns the sum y = m0x0 + m1x1 + m2x2 +....+b
    private double weightedSum(double m[], double x[], double b) {
        for (int i = 0; i < m.length; i++) {
            b += m[i] * x[i];
        }
        return b;
    }

    private double Relu(double y) {
        if (y > 0) {
            return y;
        }
        return 0;
    }

    public Neuron(double learningRate, String Method, int prevLayerNoNeurons) {
        this.Method = Method;
        this.learningRate = learningRate;
        this.weights = new double[prevLayerNoNeurons];
        if ("input".equals(this.Method)) {
            Arrays.fill(this.weights, 1); // initial value of output type neuron weights is 1
        } else {
            Random randomGenerator = new Random();
            for (int i = 0; i < prevLayerNoNeurons; i++) {
                weights[i] = randomGenerator.nextDouble();
            }
        }
        System.out.println("Created neuron with method " + this.Method + ", learning rate " + this.learningRate
                + ", weights " + Arrays.toString(this.weights));
    }

    public double output(double inputs[], Caller caller) {
        System.out.println("Calculating output for inputs: " + Arrays.toString(inputs));
        int newPrevInputSize = 1;
        if (Caller.MODEL == caller) {
            if (this.previousInputs == null) {
                this.previousInputs = new double[newPrevInputSize][inputs.length];
                this.previousInputs[newPrevInputSize - 1] = Arrays.copyOf(inputs, inputs.length);
            } else {
                newPrevInputSize = this.previousInputs.length + 1;
                double[][] temp = new double[newPrevInputSize][];
                for (int i = 0; i < newPrevInputSize - 1; i++) {
                    temp[i] = this.previousInputs[i];
                }
                temp[newPrevInputSize - 1] = Arrays.copyOf(inputs, inputs.length);
                this.previousInputs = temp;
            }
        }

        double y = weightedSum(weights, inputs, bias);
        double output = ("Relu".equals(this.Method)) ? Relu(y) : y;
        System.out.println("Neuron output: " + output);
        return output;
    }

    public void gradientDescent(double error[]) { // error=[ob1err,ob2err,ob3err,ob4err...]
        if ("input".equals(this.Method)) {
            return;
        }
        System.out.println("Performing gradient descent with error: " + Arrays.toString(error));
        int prevLayerNeuronsCount = this.weights.length;
        int numberOfObservations = this.previousInputs.length;
        System.out.println("Kya chutiyapa hai: error length->" + error.length + " ; numberOfObservations-> "
                + numberOfObservations);

        this.errorForPreviousLayer = new double[numberOfObservations][prevLayerNeuronsCount];

        for (int i = 0; i < prevLayerNeuronsCount; i++) {

            double derivative = 0;
            double biasDerivative = 0;
            double costPerItem = 0;

            for (int j = 0; j < numberOfObservations - 1; j++) {
                try {
                    costPerItem = weightedSum(this.weights, this.previousInputs[j], this.bias) - error[j];
                } catch (ArrayIndexOutOfBoundsException e) {
                    System.out.println("mai hu gian: " + j);
                }

                biasDerivative += costPerItem;
                derivative += (previousInputs[j][i] * costPerItem);

                // setting variable errorForBack for calculating backward error for giving it to
                // previous layer
                this.errorForPreviousLayer[j][i] = costPerItem;
            }

            this.bias -= (this.learningRate / numberOfObservations) * biasDerivative;
            this.weights[i] -= (this.learningRate / numberOfObservations) * derivative;

            System.out.println("Updated weight " + i + ": " + this.weights[i] + ", updated bias: " + this.bias);
        }
        this.previousInputs = null;
    }
}
