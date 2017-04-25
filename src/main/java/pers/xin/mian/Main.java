package pers.xin.mian;

import pers.xin.optimization.PSO;
import weka.classifiers.trees.J48;
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
        File file = new File("/Users/xin/workspase/DataSet/bands.arff");
        Instances instances = new Instances(new FileReader(file));
        instances.setClassIndex(instances.numAttributes()-1);
        Experiment e = new Experiment(weka.classifiers.functions.LibSVM.class.getName(),instances);
        double originalAUC = e.originalTest();
        double[][] interval = {{0,1},{0,1},{0,1}};
        e.setInterval(interval);
        System.out.println(originalAUC);
        m.analyzeStrings.add("使用分类器: "+e.getClassifierName());
        m.analyzeStrings.add("原始分类指标："+e.originalAnalyze());

        PSO pso = new PSO(10,60,0.00001,0.4,2,2);
        pso.setObject(e);
        double[] params = pso.search();
        System.out.println();
        for (double param : params) {
            System.out.print(param+"  ");
        }
        m.analyzeStrings.add("特征选择参数: "+"delta:" + params[0]+", alpha:"+params[1]+", beta:"+params[2]);
        m.analyzeStrings.add(e.RSFSAIDAnalyze(params[0],params[1],params[2]));

        System.out.println();

        System.out.println();
        System.out.println(e.RSFSAIDTest(params[0],params[1],params[2]));
        m.output();
    }

    public void output() throws Exception {
        File file = new File("/Users/xin/workspase/DataSet/bands/result.txt");
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
