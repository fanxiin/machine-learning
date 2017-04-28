package pers.xin.mian;

import pers.xin.Experiment.Experiment;
import pers.xin.Experiment.Summary;
import pers.xin.optimization.PSO;
import weka.core.Instances;

import java.io.*;
import java.util.ArrayList;

/**
 * Created by xin on 2017/4/19.
 */
public class Main {

    private ArrayList<String> analyzeStrings = new ArrayList<String>();

    public static void main(String[] args) throws Exception {

        File folder = new File("/Users/xin/Desktop/ExperimentData/myDataARFF");

        Main m = new Main();

//        String[] clssifiers = {weka.classifiers.trees.J48.class.getName()
//                ,weka.classifiers.functions.LibSVM.class.getName()};
        String[] clssifiers = {weka.classifiers.trees.J48.class.getName()
                ,weka.classifiers.trees.RandomForest.class.getName()
                ,weka.classifiers.bayes.NaiveBayes.class.getName()
                ,weka.classifiers.functions.LibSVM.class.getName()};

        File[] files = folder.listFiles();

        for (String classifierName : clssifiers) {
            m.resultPrintln(classifierName);
            m.resultPrintln(Summary.header());
            for (File file : files) {
                if(!file.getName().startsWith(".")){
                    try{
                        Instances instances = new Instances(new FileReader(file));
                        instances.setClassIndex(instances.numAttributes()-1);
                        Experiment e = new Experiment(classifierName,instances);
                        double[][] interval = {{0,0.1},{0,1},{0,1}};
                        int[] precision = {3,2,2};
                        e.setInterval(interval);
                        e.setPrecision(precision);
                        e.setInterval(interval);
                        e.setPrecision(precision);
                        Summary oSummary = e.originalAnalyze();
                        m.resultPrintln(oSummary.toString());
                        PSO pso = new PSO(20,20,1,0.00001,0.5,2,2);
                        pso.setObject(e);
                        for (int i = 0; i < 3; i++) {
                            double[] params = pso.search();
                            Summary fsSummary = e.RSFSAIDAnalyze(params);
                            m.resultPrintln(fsSummary.toString());
                            System.out.println(fsSummary.getReduction());
                            System.out.println(fsSummary.getROC_Area());
                        }

                        m.resultPrintln("");
                    }catch (Exception e){
                        e.printStackTrace();
                        continue;
                    }
                }
            }
        }


    }


//    public void output() throws Exception {
//        File file = new File("/Users/xin/workspase/DataSet/ionosphere/result.csv");
//        if(!file.getParentFile().exists()){
//            file.getParentFile().mkdirs();
//        }
//        if(!file.exists()){
//            file.createNewFile();
//        }
//        FileWriter fw = new FileWriter(file,true);
//        PrintWriter pw = new PrintWriter(fw);
//        for (String analyzeString : analyzeStrings) {
//            pw.println(analyzeString);
//        }
//        pw.close();
//    }

    public void resultPrintln(String data) throws Exception{
        File file = new File("/Users/xin/Desktop/ExperimentData/m_result/result.csv");
        if(!file.getParentFile().exists()){
            file.getParentFile().mkdirs();
        }
        if(!file.exists()){
            file.createNewFile();
        }
        FileWriter fw = new FileWriter(file,true);
        PrintWriter pw = new PrintWriter(fw);
        pw.println(data);
        pw.close();
    }
}
