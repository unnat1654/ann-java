package ArtificialNNModel;

class outputLayer extends Layer{
    outputLayer(int number,double learningRate, int prevLayerNoNeurons){
        super(number,learningRate,"output",prevLayerNoNeurons);
    }
}