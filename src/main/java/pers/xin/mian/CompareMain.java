package pers.xin.mian;

import pers.xin.Experiment.Experiment;
import pers.xin.Experiment.FormatSummary;
import pers.xin.Experiment.Summary;
import swjtu.ml.filter.supervised.WAR;
import weka.core.Instances;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;

/**
 * Created by xin on 2017/5/22.
 */
public class CompareMain {
    public static void main(String[] args) throws Exception {

        File folder = new File("/Users/xin/Desktop/ExperimentData/yeast5");

        CompareMain m = new CompareMain();
        String[] clssifiers = {weka.classifiers.trees.J48.class.getName()
                ,weka.classifiers.trees.RandomForest.class.getName()
                ,weka.classifiers.bayes.NaiveBayes.class.getName()
                ,weka.classifiers.functions.LibSVM.class.getName()};

        File[] files = folder.listFiles();
        FormatSummary.setParamNames(new String[]{"delta","alpha","beta"});
        for (String classifierName : clssifiers) {
            m.resultPrintln(classifierName);
            m.resultPrintln(FormatSummary.header());
            for (File file : files) {
                if(!file.getName().startsWith(".")){
                    System.out.println("-------- 处理数据集: "+file.getName() +" ---------");
                    try{
                        Instances instances = new Instances(new FileReader(file));
                        instances.setClassIndex(instances.numAttributes()-1);
                        Experiment e = new Experiment(classifierName,instances,5);
                        double[] param = new double[1];
//                        e.setFSAlgorithmName(FARNeM.class.getName());
//                        m.resultPrintln(e.originalAnalyze());
//                        //logger.info("-------- origin AUC"+summary.getROC_Area()+" ---------");
//                        double[] param = {0.125};
//                        m.resultPrintln(e.FSAnalyze(param));
//                        param[0] = 0.25;
//                        m.resultPrintln(e.FSAnalyze(param));
                        Summary summary;
                        summary = e.originalAnalyze();
                        m.resultPrintln(FormatSummary.format(summary));

                        e.setFSAlgorithmName(WAR.class.getName());
                        param[0] = 0.1;
                        summary = e.FSAnalyze(param);
                        m.resultPrintln(FormatSummary.format(summary));
                        param[0] = 0.05;
                        summary = e.FSAnalyze(param);
                        m.resultPrintln(FormatSummary.format(summary));
                        param[0] = 0.01;
                        summary = e.FSAnalyze(param);
                        m.resultPrintln(FormatSummary.format(summary));
                        param[0] = 0;
                        summary = e.FSAnalyze(param);
                        m.resultPrintln(FormatSummary.format(summary));
                        m.resultPrintln("");
                    }catch (Exception e){
                        e.printStackTrace();
                        continue;
                    }
                }
            }
        }
    }

    public void resultPrintln(String data) throws Exception{
        File file = new File("/Users/xin/Desktop/ExperimentData/m_result/yeast5/result_compare.csv");
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
