package ArtificialNNModel;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.IllegalArgumentException;
import java.util.Arrays;
import java.io.FileReader;
import java.io.IOException;



/**
 * The ArtificialNeuralNetwork class represents a simple artificial neural network model
 * with methods for adding layers, training, predicting, and saving weights.
 */
public class ArtificialNeuralNetwork {
    Layer levels[];
    public double outputOfNeurons[][];
    public int outputNeuronsCount = 0;
    public int inputClassesCount = 0;
    private BiMap<Double, Integer> map = new BiMap<>();

    /**
     * Default constructor initializing an empty neural network.
     */
    public ArtificialNeuralNetwork() {
        this.levels = new Layer[0];
        this.outputOfNeurons = new double[0][0];
    }

    /**
     * Constructor that initializes the neural network from a given file.
     *
     * @param filename the name of the file containing the neural network configuration
     */
    public ArtificialNeuralNetwork(String filename) {
        StringBuilder json = new StringBuilder();
        try (FileReader reader = new FileReader(filename)) {
            int ch;
            while ((ch = reader.read()) != -1) {
                json.append((char) ch);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    
        String content = json.toString();

        content = content.replace("{", "").replace("}", "").replace("[", "").replace("]", "").replace("\"", "");
        
        String[] parts = content.split(";");
        
        int layersCount = Integer.parseInt(parts[0].split(":")[1]);

        String[] neuronsParts = parts[1].split(":")[1].split(",");

        int[] neuronsCount = new int[layersCount];
        for (int i = 0; i < layersCount; i++) {
            neuronsCount[i] = Integer.parseInt(neuronsParts[i]);
        }
    
        String[] learningRatesParts = parts[2].split(":")[1].split(",");
        double[] learningRates = new double[layersCount];
        for (int i = 0; i < layersCount; i++) {
            learningRates[i] = Double.parseDouble(learningRatesParts[i]);
        }

        String[] possibleOutputsParts = parts[3].split(":")[1].split(",");
        double[] possibleOutputs = new double[possibleOutputsParts.length];
        for (int i = 0; i < possibleOutputsParts.length; i++) {
            possibleOutputs[i] = Double.parseDouble(possibleOutputsParts[i]);
        }

        for (int i = 0; i < layersCount; i++) {
            if (i == 0) {
                addInputLayer(neuronsCount[i]);
            } else if (i == layersCount - 1) {
                addOutputLayer(possibleOutputs, learningRates[i]);
            } else {
                addHiddenLayer(neuronsCount[i], learningRates[i]);
            }
        }
    
        for (int i = 0; i < layersCount; i++) {
            for (int j = 0; j < neuronsCount[i]; j++) {
                String weightsKey = "Layer_" + i + "_Neuron_" + j + ":";
                String weightsString = null;
                for (String part : parts) {
                    if (part.startsWith(weightsKey)) {
                        weightsString = part.split(":")[1];
                        break;
                    }
                }
                if (weightsString != null) {
                    String[] weightsParts = weightsString.split(",");
                    double[] weights = new double[weightsParts.length];
                    for (int k = 0; k < weightsParts.length; k++) {
                        weights[k] = Double.parseDouble(weightsParts[k]);
                    }
                    levels[i].neurons[j].weights = weights;
                }
            }
        }
    }

    /**
     * Adds an input layer to the neural network.
     *
     * @param no_of_classes the number of input neurons
     * @throws IllegalArgumentException if the number of input neurons is less than or equal to zero
     * @throws IllegalStateException if the input layer or the output layer has already been initialized
     */
    public void addInputLayer(int no_of_classes) throws IllegalArgumentException, IllegalStateException{
        if (no_of_classes <= 0) {
            throw new IllegalArgumentException("Number of input neurons must be greater than zero.");
        }
        if(this.inputClassesCount!=0 ||this.outputNeuronsCount!=0){
            throw new IllegalStateException("Input or Output Layer has already been formed");
        }
        this.levels = new Layer[1];
        this.levels[this.levels.length - 1] = new inputLayer(no_of_classes);
        this.inputClassesCount = no_of_classes;
    }

    /**
     * Adds a hidden layer to the neural network.
     *
     * @param no_of_neurons the number of neurons in the hidden layer
     * @param learningRate the learning rate for the hidden layer
     * @throws IllegalStateException if the input layer has not been initialized or the output layer is already found
     * @throws IllegalArgumentException if the number of neurons is less than or equal to zero
     */
    public void addHiddenLayer(int no_of_neurons, double learningRate)
            throws IllegalStateException, IllegalArgumentException {
        if (this.inputClassesCount == 0) {
            throw new IllegalStateException("Input Layer must be initialized before adding a hidden layer");
        }
        if(this.outputNeuronsCount!=0){
            throw new IllegalStateException("Output Layer has already been formed");
        }
        if (no_of_neurons <= 0) {
            throw new IllegalArgumentException("Number of neurons must be greater than zero.");
        }

        increaseLevelsSize();
        
        int prevLayerNoNeurons = this.levels[this.levels.length-2].numberOfNeurons;
        this.levels[this.levels.length - 1] = new Layer(no_of_neurons, learningRate, prevLayerNoNeurons);
    }
    
    /**
     * Adds an output layer to the neural network.
     *
     * @param possibleOutputs an array of possible output values
     * @param learningRate the learning rate for the output layer
     * @throws IllegalArgumentException if the number of output neurons is less than or equal to one
     * @throws IllegalStateException if the input layer has not been initialized
     */
    public void addOutputLayer(double[] possibleOutputs, double learningRate) throws IllegalArgumentException, IllegalStateException {
        if (this.inputClassesCount == 0) {
            throw new IllegalStateException("Input Layer must be initialized before adding a hidden layer");
        }
        if(this.outputNeuronsCount!=0){
            throw new IllegalStateException("Output Layer has already been formed");
        }
        if (possibleOutputs==null || possibleOutputs.length==1) {
            throw new IllegalArgumentException("Number of output neurons must be greater than one.");
        }

        for(int i=0;i<possibleOutputs.length;i++){
            if(map.containsKey(possibleOutputs[i])){
                throw new IllegalArgumentException("Matching outputs found in possible outputs.");
            }
            map.put(possibleOutputs[i],i);
        }

        increaseLevelsSize();

        int prevLayerNoNeurons = this.levels[this.levels.length-2].numberOfNeurons;
        this.outputNeuronsCount = map.size();
        this.levels[this.levels.length - 1] = new outputLayer(this.outputNeuronsCount, learningRate, prevLayerNoNeurons);
    }

    /**
     * Predicts the output for a given input array.
     *
     * @param input the input array
     * @return the predicted output value
     * @throws IllegalArgumentException if the input array length does not match the number of input neurons
     * @throws IllegalStateException if the network is not properly initialized with input and output layers
     */
    public double predict(double input[]) throws IllegalArgumentException, IllegalStateException {
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
        return map.getKey(arrayMaxIndex(levels[levels.length - 1].neuronOutputs));
        
    }

    /**
     * Calculates the accuracy of the neural network for given inputs and outputs.
     *
     * @param inputs an array of input arrays
     * @param outputs an array of expected output values
     * @return the accuracy of the model
     * @throws IllegalArgumentException if the number of inputs does not match the number of outputs
     * @throws IllegalStateException if the network is not properly initialized with input and output layers
     */
    public double accuracy(double inputs[][], double outputs[]) throws IllegalArgumentException, IllegalStateException {
        if (this.outputNeuronsCount == 0) {
            throw new IllegalStateException(
                    "The neural network layers not set up properly. Input and output layers are required.");
        }
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("Number of observations and labels do not match");
        }
        int correctAnswers = 0;
       
        for (int i = 0; i < inputs.length; i++) {
            double prediction = this.predict(inputs[i]);
            if (map.getValue(outputs[i]) == null) {
                System.err.println(outputs[i]);
                throw new IllegalArgumentException("Output does not match labels");
            }
            if (prediction == outputs[i])
                correctAnswers++;
        }
        double accuracy = (double) correctAnswers / outputs.length;
        return accuracy;
    }

    /**
     * Trains the neural network with given inputs and outputs for a specified number of epochs.
     *
     * @param inputs an array of input arrays
     * @param outputs an array of expected output values
     * @param epochs the number of training iterations
     * @throws IllegalArgumentException if the inputs or outputs are invalid
     * @throws IllegalStateException if the network is not properly initialized with input and output layers
     */
    public void fit(double inputs[][], double outputs[], int epochs) throws IllegalArgumentException {
        if (this.outputNeuronsCount == 0) {
            throw new IllegalStateException("The neural network layers not set up properly. Input and output layers are required.");
        }
        if (inputs[0].length != levels[0].numberOfNeurons || inputs.length != outputs.length) {
            throw new IllegalArgumentException("No of classes in input are not valid");
        }
        this.outputOfNeurons = new double[this.outputNeuronsCount][outputs.length];

        int i = 0;
        for (double output : outputs) {
            if (!map.containsKey(output)) {
                throw new IllegalArgumentException("The output array contains more unique outputs than provided in NoOutputNeurons in output Layer initiailization.");
            }
        }
        for (i = 0; i < outputs.length; i++) {
            this.outputOfNeurons[map.getValue(outputs[i])][i] = 1;
        }
        // map.keyToValue.forEach((k, v) -> {
        //     System.out.println(k + " -> " + v);
        // });

        while (0 < epochs--) {
            System.out.print("Epochs left: " + (epochs+1));
            double accuracy = forwardPropogation_accuracy(inputs, outputs);
            System.out.println("\tModel Accuracy: " + accuracy);
            backPropogation();
        }
    }

    /**
     * Saves the current weights of the neural network to a file.
     *
     * @param filename the name of the file to save the weights to
     */
    public void saveWeights(String filename) {
        StringBuilder json = new StringBuilder();
        json.append("{");
        json.append("\"LayersCount\":").append(levels.length).append(";");

        json.append("\"NeuronsCount\":[");
        for (int i = 0; i < levels.length; i++) {
            json.append(levels[i].neurons.length);
            if (i < levels.length - 1) {
                json.append(",");
            }
        }
        json.append("];");

        json.append("\"LearningRates\":[");
        for (int i = 0; i < levels.length; i++) {
            json.append(levels[i].learningRate);
            if (i < levels.length - 1) {
                json.append(",");
            }
        }
        json.append("];");


        double[] possibleOutputs=new double[map.size()];
        map.keyToValue.forEach((key,value)->{
            possibleOutputs[value]=key;
        });

        json.append("\"PossibleOutputs\":");
        json.append(arrayToJson(possibleOutputs));
        json.append(";");

        
        for (int i = 0; i < levels.length; i++) {
            for (int j = 0; j < levels[i].neurons.length; j++) {
                json.append("\"Layer_").append(i).append("_Neuron_").append(j).append("\":");
                json.append(arrayToJson(levels[i].neurons[j].weights));
                json.append(";");
            }
        }

        json.append("}");

        try (FileWriter file = new FileWriter(filename)) {
            file.write(json.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void increaseLevelsSize(){
        Layer[] temp = this.levels;
        this.levels = new Layer[temp.length + 1];
        for (int i = 0; i < temp.length; i++) {
            this.levels[i] = temp[i];
        }
    }

    private String arrayToJson(double[] array) {
        StringBuilder json = new StringBuilder();
        json.append("[");
        for (int i = 0; i < array.length; i++) {
            json.append(array[i]);
            if (i < array.length - 1) {
                json.append(",");
            }
        }
        json.append("]");
        return json.toString();
    }

    private void backPropogation() {
        this.levels[levels.length - 1].receiveFrontError(this.outputOfNeurons);
        for (int i = levels.length - 1; i >= 2; i--) {
            this.levels[i - 1].receiveFrontError(levels[i].sendBackError());
        }
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
        return index;
    }

    private double[] comp_predict(double input[]) throws IllegalArgumentException, IllegalStateException {
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
        return levels[levels.length - 1].neuronOutputs;
    }

    private double forwardPropogation_accuracy(double inputs[][], double outputs[]) throws IllegalArgumentException, IllegalStateException {
        if (this.outputNeuronsCount == 0) {
            throw new IllegalStateException(
                    "The neural network layers not set up properly. Input and output layers are required.");
        }
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("Number of observations and labels do not match");
        }
        int correctAnswers = 0;
       
        for (int i = 0; i < inputs.length; i++) {
            int predictionIndex = arrayMaxIndex(this.comp_predict(inputs[i]));
            if (map.getValue(outputs[i]) == null) {
                System.err.println(outputs[i]);
                throw new IllegalArgumentException("Output does not match labels");
            }
            if (predictionIndex == map.getValue(outputs[i]))
                correctAnswers++;
        }
        double accuracy = (double) correctAnswers / outputs.length;

        return accuracy;
            
    }

}

enum Caller {
    USER,
    MODEL
}