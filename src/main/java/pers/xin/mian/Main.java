package pers.xin.mian;

import pers.xin.Experiment.Experiment;
import pers.xin.Experiment.Summary;
import pers.xin.optimization.PSO;
import weka.core.Instances;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * Created by xin on 2017/4/19.
 */
public class Main {

    private ArrayList<String> analyzeStrings = new ArrayList<String>();

    public static void main(String[] args) throws Exception {
        Main m = new Main();
        File file = new File("/Users/xin/workspase/DataSet/ionosphere.arff");
        Instances instances = new Instances(new FileReader(file));
        instances.setClassIndex(instances.numAttributes()-1);
        Experiment e = new Experiment(weka.classifiers.trees.J48.class.getName(),instances);
        double[][] interval = {{0,0.1},{0,1},{0,1}};
        e.setInterval(interval);
        Summary oSummary = e.originalAnalyze();
        System.out.println(oSummary.getROC_Area());
        PSO pso = new PSO(10,60,1,0.00001,1,2,2);
        pso.setObject(e);
        double[] params = pso.search();
        Summary fsSummary = e.RSFSAIDAnalyze(params);
        System.out.println(fsSummary.getReduction());
        System.out.println(fsSummary.getROC_Area());
        m.analyzeStrings.add(e.getClassifierName());
        m.analyzeStrings.add(oSummary.header());
        m.analyzeStrings.add(oSummary.toString());
        m.analyzeStrings.add(fsSummary.toString());
        m.output();
    }

    public void output() throws Exception {
        File file = new File("/Users/xin/workspase/DataSet/ionosphere/result.csv");
        if(!file.getParentFile().exists()){
            file.getParentFile().mkdirs();
        }
        if(!file.exists()){
            file.createNewFile();
        }
        FileWriter fw = new FileWriter(file,true);
        PrintWriter pw = new PrintWriter(fw);
        for (String analyzeString : analyzeStrings) {
            pw.println(analyzeString);
        }
        pw.close();
    }
}
