package ArtificialNNModel;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.lang.IllegalArgumentException;

public class ArtificialNeuralNetwork {
    Layer levels[];
    public double outputOfNeurons[][];
    public int outputNeuronsCount = 0;
    public int inputClassesCount = 0;
    private String type;
    private HashMap<Double, Integer> map = new HashMap<>();

    private void backPropogation() {
        this.levels[levels.length - 1].receiveFrontError(this.outputOfNeurons);
        for (int i = levels.length - 1; i >= 2; i--) {// lower bound is 2 since input layer does not need to adjust weights
            this.levels[i - 1].receiveFrontError(levels[i].sendBackError());
        }
    }
    
    private int arrayMaxIndex(double[] array) {
        double max = 0;
        int index = 0;
        for (int i = 0; i < array.length; i++) {
            if(array[i] > max){
                max =array[i];
                index=i;
            }
        }
        return index;
    }

    /**
     * Adds an input layer to the model with the specified number of classes/features.
     *
     * @param no_of_classes The number of classes or features in the input layer. This determines the size of the input layer.
     * @throws IllegalArgumentException If the number of classes is less than or equal to zero.
     */
    public void addInputLayer(int no_of_classes) throws IllegalArgumentException{
        if (no_of_classes <= 0) {
            throw new IllegalArgumentException("Number of input neurons must be greater than zero.");
        }
        this.levels = new Layer[1];
        this.levels[this.levels.length - 1] = new inputLayer(no_of_classes);
        this.inputClassesCount=no_of_classes;
    }

    /**
     * Adds a hidden layer to the model with the specified number of neurons and learning rate.
     *
     * @param no_of_neurons The number of neurons in this hidden layer. This determines the size of the layer.
     * @param learningRate The learning rate for the neurons in this hidden layer, which influences the adjustment of weights during training.
     * @throws IllegalStateException If the input layer has not been initialized before adding a hidden layer.
     * @throws IllegalArgumentException If the number of neurons is less than or equal to zero.
     */
    public void addHiddenLayer(int no_of_neurons, double learningRate) throws IllegalStateException, IllegalArgumentException {
        if(this.inputClassesCount==0){
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

    /**
     * Adds an output layer to the model with the specified configuration.
     *
     * @param type The type of output layer, which can be either "classification" or "regression". 
     *             This determines the nature of the output values and the loss function used.
     * @param outputNeuronsCount The number of neurons in the output layer, which corresponds to the number of output values or classes.
     * @param learningRate The learning rate for the output layer, which influences how weights are adjusted during training.
     * @throws IllegalStateException If the input layer has not been initialized before adding the output layer.
     * @throws IllegalArgumentException If the provided type is neither "classification" nor "regression", or if the number of output neurons is less than or equal to zero.
     */
    public void addOutputLayer(String type, int outputNeuronsCount, double learningRate) throws IllegalArgumentException, IllegalStateException{
        if(this.inputClassesCount==0){
            throw new IllegalStateException("Input Layer must be initialized before adding a hidden layer");
        }
        if (!"regression".equals(type) && !"classification".equals(type)) {
            throw new IllegalArgumentException("The 'type' property can be 'classification' or 'regression' only.");
        }
        if (outputNeuronsCount <= 0) {
            throw new IllegalArgumentException("Number of output neurons must be greater than zero.");
        }

        //increase levels size to add output layer
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
        this.outputNeuronsCount = Math.max(outputNeuronsCount,1);
        this.levels[this.levels.length - 1] = new Layer(this.outputNeuronsCount, learningRate, prevLayerNoNeurons);
    }
    
    /**
     * Predicts the output for a given input observation.
     *
     * @param input An array representing a single observation. Each element in the array corresponds to a feature or class of the observation.
     * @return An array of predicted output values for the given input observation.
     * @throws IllegalArgumentException if the length of the input array does not match the number of classes in the input layer.
     * @throws IllegalStateException if the neural network layers are not properly initialized (i.e., the output layer is missing).
     */
    public double[] predict(double input[]) throws IllegalArgumentException, IllegalStateException {
        if(outputNeuronsCount==0){
            throw new IllegalStateException("The neural network layers not set up properly. Input and output layers are required.");
        }
        if(input.length!=inputClassesCount){
            throw new IllegalArgumentException("The numbers of classes in input layer do not match classes in entered array.");
        }
        levels[0].calculateOutputs(input);
        for (int i = 1; i < levels.length; i++) {
            levels[i].calculateOutputs(levels[i - 1].neuronOutputs);
        }
        return levels[levels.length-1].neuronOutputs;
    }

    /**
     * Calculates the accuracy of the model's predictions based on the provided inputs and expected outputs.
     *
     * @param inputs  A 2D array where each row represents a single observation with feature values.
     *                Each row in this array corresponds to an observation for which predictions are made.
     *                The structure is as follows:
     *                <pre>
     *                [
     *                  [observation1feature1, observation1feature2, ...],
     *                  [observation2feature1, observation2feature2, ...],
     *                  ...
     *                ]
     *                </pre>
     * @param outputs An array of expected output values corresponding to each observation in `inputs`.
     *                The length of this array must match the number of rows in `inputs`.
     * @return The accuracy of the model as a percentage, calculated as the proportion of correct predictions
     *         to the total number of observations. For regression, returns the mean squared error.
     * @throws IllegalArgumentException if the number of observations in `inputs` does not match the length of `outputs`.
     * @throws IllegalStateException if the neural network layers are not properly initialized (i.e., input and output layers are required).
     */
    public double accuracy(double inputs[][], double outputs[]) throws IllegalArgumentException, IllegalStateException{
        if(this.outputNeuronsCount==0){
            throw new IllegalStateException("The neural network layers not set up properly. Input and output layers are required.");
        }
        if(inputs.length!=outputs.length){
            throw new IllegalArgumentException("Number of observations and labels do not match");
        }
        int correctAnswers = 0;
        if ("classification".equals(this.type)) {
            for (int i = 0; i < inputs.length; i++) {
                int predictionIndex = arrayMaxIndex(this.predict(inputs[i]));
                if(map.get(outputs[i])==null){
                    throw new IllegalArgumentException("Output does not match labels");
                }
                if (predictionIndex == map.get(outputs[i]))
                    correctAnswers++;
            }
            return correctAnswers / outputs.length;
        } 
        // else if ("regression".equals(this.type)) {
        double error = 0;
        for (int i = 0; i < inputs.length; i++) {
            double prediction = this.predict(inputs[i])[0];
            error += Math.pow(outputs[i] - prediction, 2);
        }
        return error / outputs.length;
        // }
        // return 0;
    }

    /**
     * Fits the model to the given inputs and outputs for a specified number of epochs.
     *
     * @param inputs  A 2D array where each element is an observation. 
     *                Each observation is represented as an array of classes.
     *                The structure is as follows:
     *                <pre>
     *                [
     *                  [observation1class1, observation1class2, observation1class3, ...],
     *                  [observation2class1, observation2class2, observation2class3, ...],
     *                  ...
     *                ]
     *                </pre>
     * @param outputs An array of expected output values corresponding to each observation.
     * @param epochs  The number of epochs to train the model.
     * @throws IllegalArgumentException if:
     *                                  - The number of classes in the input data does not match the number of neurons in the input layer.
     *                                  - The length of the `outputs` array does not match the number of rows in `inputs`.
     *  @throws IllegalStateException if the neural network layers are not set up properly (i.e., input and output layers are not initialized).
     */    
    public void fit(double inputs[][], double outputs[], int epochs) throws IllegalArgumentException {
        if(this.outputNeuronsCount==0){
            throw new IllegalStateException("The neural network layers not set up properly. Input and output layers are required.");
        }
        if (inputs[0].length != levels[0].numberOfNeurons || inputs.length!=outputs.length) {
            throw new IllegalArgumentException("No of classes in input are not valid");
        }
        this.outputOfNeurons = new double[this.outputNeuronsCount][outputs.length];
        if ("regression".equals(this.type)) {
            this.outputOfNeurons[0] = outputs;// [[n1ob1,n1ob2,n1ob3,n1ob4,n2ob5...]]
        }
        if ("classification".equals(this.type)) {
            int i = 0;
            // getting the mapping for encoding
            for (double output : outputs) {
                if (map.size() == this.outputNeuronsCount)
                    break;
                if (!map.containsKey(output)) {
                    if (map.size() < this.outputNeuronsCount)
                        map.put(output, i++);
                    else
                        throw new IllegalArgumentException("The output array contains more unique outputs than provided in NoOutputNeurons in output Layer initiailization.");
                }
            }
            // encoding the outputs according to the map
            for (i = 0; i < outputs.length; i++) {
                // [[n1ob1,n1ob2,n1ob3,n1ob4...],[n2o1,n2o2...]]
                this.outputOfNeurons[map.get(outputs[i])][i] = 1;
            }
            map.forEach((k,v)->{
                System.out.println(k+" -> "+v);
            });

        }

        // Fitting the inputs and outputs to algorithm
        while (0 < epochs--) {
            System.out.print("Epochs passed: " + epochs);
            double acracy = accuracy(inputs, outputs);
            System.out.println("Model Accuracy:" + acracy);
            // accuracy function passes the values of input and output to train the model
            backPropogation();
        }
    }

}

class Layer {
    protected Neuron neurons[];
    private int prevLayerNeuronsCount=1;

    private void generateNeurons() {
        for (int i = 0; i < this.numberOfNeurons; i++) {
            this.neurons[i] = new Neuron(this.learningRate, this.Method, this.prevLayerNeuronsCount);
        }
    }


    public String Method = "Relu";// default
    public double learningRate = 0.01;// default
    public int numberOfNeurons = 1;// default
    public double neuronOutputs[];
    
    //Constructors
    public Layer(int number, double learningRate, int prevLayerNoNeurons) {
        this.neurons = new Neuron[number];
        this.numberOfNeurons=number;
        this.learningRate = learningRate;
        this.prevLayerNeuronsCount=prevLayerNoNeurons;
        this.neuronOutputs=new double[number];
        generateNeurons();
    }
    public Layer(int number, double learningRate, String Method, int prevLayerNoNeurons) {
        this.neurons = new Neuron[number];
        this.numberOfNeurons=number;
        this.learningRate = learningRate;
        this.Method = Method;
        this.neuronOutputs=new double[number];
        this.prevLayerNeuronsCount=prevLayerNoNeurons;
        generateNeurons();
    }
    
    public void calculateOutputs(double inputs[]) {// inputs=[cl1,cl2,cl3,cl4,cl5...]
        for (int i = 0; i < this.numberOfNeurons; i++) {
            this.neuronOutputs[i] = this.neurons[i].output(inputs);
        }
    }

    public void receiveFrontError(double error[][]) {// error=[thisLayerneuronnumber][DataItemsNo] or [[n1o1,n1o2,n1o3,n1o4,n1o5...],[n2o1,n2o2,n2o3...]..]
        for (int i = 0; i < this.numberOfNeurons; i++) {
            this.neurons[i].gradientDescent(error[i]);
        }
    }

    public double[][] sendBackError() {
        int numberOfObservations=this.neurons[0].errorForPreviousLayer.length;
        int prevLayerNeuronsCount=this.neurons[0].errorForPreviousLayer[0].length;

        double[][][] thisLayerError = new double[numberOfNeurons][numberOfObservations][prevLayerNeuronsCount];
        for (int i = 0; i < neurons.length; i++) {
            thisLayerError[i] = neurons[i].errorForPreviousLayer;
        }

        double[][] addedThisLayerError = new double[prevLayerNeuronsCount][numberOfObservations];
        for (int i = 0; i < prevLayerNeuronsCount; i++) {
            for (int j = 0; j < numberOfObservations; j++) {
                for (int k = 0; k < numberOfNeurons; k++) {
                    addedThisLayerError[i][j] += this.neurons[k].errorForPreviousLayer[j][i];
                }
            }
        }

        return addedThisLayerError;
    }

}

class inputLayer extends Layer {
    public inputLayer(int number) {
        super(number, 0, "input", 1);
    }

    @Override
    public void calculateOutputs(double inputs[]) { // the input is [cl1,cl2,cl3,cl4,cl5....]
        for (int i = 0; i < super.numberOfNeurons; i++) {
            double[] input = { inputs[i] }; // input=[cl1]
            super.neuronOutputs[i] = super.neurons[i].output(input);
        }
    }
}

class Neuron {
    private double weights[];
    private double bias = 0;
    private double learningRate;
    private double previousInputs[][];// previousInputs[DataItemsNo][PrevoiusLayerNeuronNumber]
    public double errorForPreviousLayer[][];// errorForBack=[[ob1weighterror1,ob1we2,ob1we3...],[ob2we1,ob2we2...]...]
    public String Method = "Relu";// enum: "input" and "Relu"

    // returns the sum y = m0x0 + m1x1 + m2x2 +....+b
    private double weightedSum(double m[], double x[], double b) {
        for (int i = 0; i < m.length; i++) {
            b += m[i] + x[i];
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
        if("input"==this.Method){
            Arrays.fill(this.weights, 1);// initial value of output type neuron weights is 1
        }
        else {
            Random randomGenerator = new Random();
            for(int i=0;i<prevLayerNoNeurons;i++){
                weights[i]=randomGenerator.nextDouble();
            }
        }
    }

    public double output(double inputs[]) {

        int newPrevInputSize=1;
        if(this.previousInputs==null){
            this.previousInputs=new double[newPrevInputSize][inputs.length];
            this.previousInputs[newPrevInputSize-1]=Arrays.copyOf(inputs, inputs.length);
        } else {
            newPrevInputSize=this.previousInputs.length;
            double[][] temp=new double[newPrevInputSize][];
            for (int i=0;i<newPrevInputSize;i++) {
                temp[i]=this.previousInputs[i];
            }
            temp[newPrevInputSize-1]=Arrays.copyOf(inputs, inputs.length);
            this.previousInputs=temp;
        }
        
        double y = weightedSum(weights, inputs, bias);
        if ("Relu".equals(this.Method)) return Relu(y);
        return y;
    }

    public void gradientDescent(double error[]) {// error-[ob1err,ob2err,ob3err,ob4err...]
        if("input".equals(this.Method)){
            return;
        }
        int prevLayerNeuronsCount=this.weights.length;
        int numberOfObservations=this.previousInputs.length;

        this.errorForPreviousLayer=new double[numberOfObservations][prevLayerNeuronsCount];

        for (int i = 0; i < prevLayerNeuronsCount; i++) {

            double derivative = 0;
            double biasDerivative = 0;
            double costPerItem = 0;

            for (int j = 0; j < numberOfObservations; j++) {

                costPerItem = (weightedSum(this.weights, this.previousInputs[j], this.bias) - error[j]);

                biasDerivative += costPerItem;
                derivative += (previousInputs[j][i] * costPerItem);

                // setting variable errorForBack for calculating backward error for giving it to previous layer
                this.errorForPreviousLayer[j][i] = costPerItem;
            }

            this.bias -= (this.learningRate / numberOfObservations) * biasDerivative;
            this.weights[i] -= (this.learningRate / numberOfObservations) * derivative;
        }
    }
}




