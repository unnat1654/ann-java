package ArtificialNNModel;

import java.util.Arrays;
import java.util.stream.IntStream;

class Layer {
    private int prevLayerNeuronsCount = 1;
    
    Neuron neurons[];
    String Method = "sigmoid"; // default
    double learningRate = 0.01; // default
    int numberOfNeurons = 1; // default
    double neuronOutputs[];

    

    private void generateNeurons() {
        for (int i = 0; i < this.numberOfNeurons; i++) {
            this.neurons[i] = new Neuron(this.learningRate, this.Method, this.prevLayerNeuronsCount);
        }
    }

    

    // Constructors
    Layer(int number, double learningRate, int prevLayerNoNeurons) {
        this.neurons = new Neuron[number];
        this.numberOfNeurons = number;
        this.learningRate = learningRate;
        this.prevLayerNeuronsCount = prevLayerNoNeurons;
        this.neuronOutputs = new double[number];
        generateNeurons();
    }

    Layer(int number, double learningRate, String Method, int prevLayerNoNeurons) {
        this.neurons = new Neuron[number];
        this.numberOfNeurons = number;
        this.learningRate = learningRate;
        this.Method = Method;
        this.neuronOutputs = new double[number];
        this.prevLayerNeuronsCount = prevLayerNoNeurons;
        generateNeurons();
    }

    void calculateOutputs(double inputs[], Caller caller) { // inputs=[cl1,cl2,cl3,cl4,cl5...]
        Arrays.parallelSetAll(this.neuronOutputs, i->this.neurons[i].output(inputs, caller));
    }

    void receiveFrontError(double error[][]) { // error=[thisLayerneuronnumber][DataItemsNo] or [[n1o1,n1o2,n1o3,n1o4,n1o5...],[n2o1,n2o2,n2o3...]..]
        IntStream.range(0, neurons.length).parallel().forEach(i -> neurons[i].gradientDescent(error[i]));
    }

    double[][] sendBackError() {
        int numberOfObservations = this.neurons[0].errorForPreviousLayer.length;
        int prevLayerNeuronsCount = this.neurons[0].errorForPreviousLayer[0].length;

        double[][] addedThisLayerError = new double[prevLayerNeuronsCount][numberOfObservations];

        IntStream.range(0,addedThisLayerError.length).parallel().forEach(i->
            IntStream.range(0, addedThisLayerError[i].length).parallel().forEach(j->{
                for(int k=0;k<this.numberOfNeurons;k++){
                    addedThisLayerError[i][j]+=this.neurons[k].errorForPreviousLayer[j][i];
                }
                for(Neuron neuron: this.neurons){
                    addedThisLayerError[i][j]+=neuron.errorForPreviousLayer[j][i];
                }
            })
        );
        for(Neuron neuron: this.neurons){
            neuron.errorForPreviousLayer=null;
        }

        return addedThisLayerError;
    }
}