package ArtificialNNModel;

import java.util.Arrays;
import java.util.Random;

class Neuron {
    double weights[];
    private double learningRate;
    private double previousInputs[][]; // previousInputs[DataItemsNo][PrevoiusLayerNeuronNumber]
    public double errorForPreviousLayer[][]; // errorForBack=[[ob1weighterror1,ob1we2,ob1we3...],[ob2we1,ob2we2...]...]
    public String Method = "sigmoid"; // enum: "input" and "sigmoid" and "output"

    // returns the sum y = m0x0 + m1x1 + m2x2 +....+b
    private double weightedSum(double m[], double x[]) {
        double sum=0;
        for (int i = 0; i < m.length; i++) {
            sum += m[i] * x[i];
        }
        return sum;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x)); // Sigmoid function formula
    }

    Neuron(double learningRate, String Method, int prevLayerNoNeurons) {
        this.Method = Method;
        this.learningRate = learningRate;
        this.weights = new double[prevLayerNoNeurons];
        if ("input".equals(this.Method)) {
            Arrays.fill(this.weights, 1); // initial value of output type neuron weights is 1
        } else {
            Random randomGenerator = new Random();
            for (int i = 0; i < prevLayerNoNeurons; i++) {
                weights[i] = randomGenerator.nextDouble()-0.5;
            }
        }
    }

    double output(double inputs[], Caller caller) {
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
                temp=null;
            }
        }

        double y = weightedSum(weights, inputs);
        double output;
        if("input".equals(this.Method)){
            output=y;
        }
        else {
            return sigmoid(y);
        }
        return output;
    }

    void gradientDescent(double expectedOutputs[]) { // expectedOutputs=[ob1err,ob2err,ob3err,ob4err...]
        if ("input".equals(this.Method)) {
            return;
        }
        int prevLayerNeuronsCount = this.weights.length;
        int numberOfObservations = this.previousInputs.length;
    
        this.errorForPreviousLayer = new double[numberOfObservations][prevLayerNeuronsCount];
        for (int j = 0; j < numberOfObservations; j++) {
    
            for (int i = 0; i < prevLayerNeuronsCount; i++) {
                double costPerItem = 0;
    
                if ("output".equals(this.Method)) {
                    costPerItem = sigmoid(weightedSum(this.weights, this.previousInputs[j])) - expectedOutputs[j];
                }
                if ("sigmoid".equals(this.Method)) {
                    costPerItem = sigmoid(weightedSum(this.weights, this.previousInputs[j])) * expectedOutputs[j];
                }
    
                // setting variable errorForBack for calculating backward error for giving it to previous layer
                this.errorForPreviousLayer[j][i] = costPerItem;
    
                this.weights[i] -= this.learningRate * costPerItem * previousInputs[j][i];
            }
        }
        this.previousInputs = null;
    }
}