package pers.xin.mian;

import org.apache.log4j.Logger;
import pers.xin.Experiment.Experiment;
import pers.xin.Experiment.FormatSummary;
import pers.xin.optimization.PSO;
import swjtu.ml.filter.supervised.FARNeM;
import swjtu.ml.filter.supervised.RSFSAID;
import swjtu.ml.filter.supervised.RSFSAIDS;
import swjtu.ml.filter.supervised.WAR;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by xin on 2017/4/19.
 */
public class Main {
    private static Logger logger=Logger.getLogger("detail");
    private static Logger m_logger=Logger.getLogger("matlab");

    public static void main(String[] args) throws Exception {

        File folder = new File("/Users/xin/Desktop/ExperimentData/RSFSAIDS");

        Main m = new Main();

//        String[] clssifiers = {weka.classifiers.trees.J48.class.getName()
//                ,weka.classifiers.functions.LibSVM.class.getName()};
        String[] clssifiers = {weka.classifiers.trees.J48.class.getName()
                ,weka.classifiers.trees.RandomForest.class.getName()
                ,weka.classifiers.bayes.NaiveBayes.class.getName()
                ,weka.classifiers.functions.LibSVM.class.getName()};

        int maxIterate=20;
        int swarmSize=20;
        int psoTimes=1;

        File[] files = folder.listFiles();

        StringBuilder ss = new StringBuilder();
        for (String clssifier : clssifiers) {
            ss.append(clssifier);
            ss.append("\n");
        }
        ss.append("----------------\n");
        for (File file : files) {
            String name = file.getName();
            if(!name.startsWith(".")){
                ss.append(name);
                ss.append("\n");
            }
        }
        ss.append("swarmSize:"+swarmSize+"\n");
        ss.append("MaxIterate:"+maxIterate+"\n");
        ss.append("PSOTimes:"+psoTimes+"\n");

        ss.append("----------------\n");
        m_logger.info(ss.toString());
        logger.info(ss.toString());
//
        FormatSummary summary = new FormatSummary("",3);
        double[][] interval = {{0,0.1},{0,1},{0,1}};
        int[] precision = {3,2,2};

//        FormatSummary summary = new FormatSummary("",1);
//        double[][] interval = {{0,0.1}};
//        int[] precision = {2};
//        HashMap<String,Double> weight = new HashMap<String, Double>();
//        weight.put("positive",30.0);
//        weight.put("negative",1.0);

        for (String classifierName : clssifiers) {
            m.resultPrintln(classifierName);
            m.resultPrintln(summary.header());
            for (File file : files) {
                if(!file.getName().startsWith(".")){
                    System.out.println("-------- 处理数据集: "+file.getName() +" ---------");
                    logger.info("-------- "+file.getName() +"-"+classifierName+" ---------");

                    try{
                        Instances instances = new Instances(new FileReader(file));
                        instances.setClassIndex(instances.numAttributes()-1);
//                        Instances t = m.smote(instances);
                        Experiment e = new Experiment(classifierName,instances,5,summary);
                        e.setFSAlgorithmName(RSFSAIDS.class.getName());
                        e.setInterval(interval);
                        e.setPrecision(precision);
                        m.resultPrintln(e.originalAnalyze());
                        //logger.info("-------- origin AUC"+summary.getROC_Area()+" ---------");
                        PSO pso = new PSO(swarmSize,maxIterate,1,0.00001,0.5,2,2);
                        pso.setObject(e);
                        for (int i = 0; i < psoTimes; i++) {
                            logger.info("-------- pso"+i+" ---------");
                            double[] params = pso.search();
                            m.resultPrintln(e.FSAnalyze(params));
//                            System.out.println(fsSummary.getReduction());
//                            System.out.println(fsSummary.getROC_Area());
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

    public Instances smote(Instances instances) throws Exception {
        int count[] = new int[2];
        for (Instance instance : instances) {
            count[(int)instance.classValue()]++;
        }
        int percentage = count[0]>count[1] ? (count[0]-count[1])*100/count[1] : (count[1]-count[0])*100/count[0];
        SMOTE s = new SMOTE();
        int seed = (int) (Math.random() * 10);
        String[] options = {"-S", String.valueOf(seed), "-P", ""+percentage, "-K", "5"};
        s.setOptions(options);
        s.setInputFormat(instances);
        return Filter.useFilter(instances,s);
    }


    public void resultPrintln(String data) throws Exception{
        File file = new File("/Users/xin/Desktop/ExperimentData/RSFSAIDSTest/result.csv");
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
