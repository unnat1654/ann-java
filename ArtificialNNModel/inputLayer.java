package ArtificialNNModel;

import java.util.Arrays;

class inputLayer extends Layer {
    inputLayer(int number) {
        super(number, 0, "input", 1);
    }

    @Override
    void calculateOutputs(double inputs[], Caller caller) { // the input is [cl1,cl2,cl3,cl4,cl5....]
        Arrays.parallelSetAll(super.neuronOutputs, i->super.neurons[i].output(new double[]{inputs[i]}, caller));
    }
}