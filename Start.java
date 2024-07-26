import java.util.List;
import ReadCSV.CSVReader;
import ArtificialNNModel.ArtificialNeuralNetwork;

public class Start {
    public static void main(String[] args) {
        List<String[]> myData=CSVReader.readCSV("cwinequality.csv");
        final float SPLIT=0.8f;
        int dataSize=myData.size();
        int number_of_features=myData.get(0).length-1;

        int trainAmt=(int) Math.floor(SPLIT*(float)dataSize);

        double[][] xTrain=new double[trainAmt][number_of_features];
        double[][] xTest= new double[myData.size()-trainAmt][number_of_features];
        double[] yTrain=new double[trainAmt];
        double[] yTest=new double[myData.size()-trainAmt];

        try{
            for(int i=0;i<dataSize;i++){
                if(i<trainAmt){
                    for(int j=0;j<number_of_features;j++){
                        xTrain[i][j]=Double.parseDouble(myData.get(i)[j]);
                        
                    }
                    yTrain[i]=Double.parseDouble(myData.get(i)[number_of_features]);
                } else{
                    for(int j=0;j<number_of_features;j++){
                        xTest[i-trainAmt][j]=Double.parseDouble(myData.get(i)[j]);
                    }
                    yTest[i-trainAmt]=Double.parseDouble(myData.get(i)[number_of_features]);
                }
            }
        } catch (NumberFormatException e) {
            System.out.println("Invalid input found");
        }

        ArtificialNeuralNetwork ANNtest=new ArtificialNeuralNetwork();
        ANNtest.addInputLayer(12);
        ANNtest.addHiddenLayer(10, 0.1);
        ANNtest.addHiddenLayer(11, 0.01);
        ANNtest.addOutputLayer("classification", 9,0.001);
        ANNtest.fit(xTrain, yTrain, 15);
        double[] ans=ANNtest.predict(xTest[1]);
        for(int i=0;i<9;i++){
            System.out.print(ans[i]+",");
        }
        System.out.println("");
        System.out.println(yTest[1]);

    }
}

