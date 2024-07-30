package ArtificialNNModel;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CSVReader {
    public static List<String[]> readCSV(String filePath) {
        List<String[]> rows = new ArrayList<>();
        String line;
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                rows.add(values);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return rows;
    }

    public static void writeHead(List<String[]> data, int rows) {
        int toPrint = Math.min(data.size(), rows);
        for (int i = 0; i < toPrint; i++) {
            for (int j = 0; j < data.get(i).length; j++) {
                System.out.print(data.get(i)[j] + ",");
            }
            System.out.println("");
        }
    }

    @SuppressWarnings({ "rawtypes", "unchecked" })
    public static ArrayList trainTestSplit(List<String[]> data, float split){
        if(split<0 || split>1){
            split=0.8f;
        }
        int dataSize=data.size();
        int number_of_features=data.get(0).length-1;

        int trainAmt=(int) Math.floor(split*(float)dataSize);

        double[][] xTrain=new double[trainAmt][number_of_features];
        double[][] xTest= new double[data.size()-trainAmt][number_of_features];
        double[] yTrain=new double[trainAmt];
        double[] yTest=new double[data.size()-trainAmt];

        try{
            for(int i=0;i<dataSize;i++){
                if(i<trainAmt){
                    for(int j=0;j<number_of_features;j++){
                        xTrain[i][j]=Double.parseDouble(data.get(i)[j]);
                        
                    }
                    yTrain[i]=Double.parseDouble(data.get(i)[number_of_features]);
                } else{
                    for(int j=0;j<number_of_features;j++){
                        xTest[i-trainAmt][j]=Double.parseDouble(data.get(i)[j]);
                    }
                    yTest[i]=Double.parseDouble(data.get(i)[number_of_features]);
                }
            }
        } catch (NumberFormatException e) {
            System.out.println("Invalid input found");
        }

        ArrayList splitData=new ArrayList();

        splitData.add(xTrain);
        splitData.add(xTest);
        splitData.add(yTrain);
        splitData.add(yTest);

        return splitData;

    }
}
