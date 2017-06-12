package pers.xin.test;

import swjtu.ml.filter.FSException;
import swjtu.ml.filter.FeatureSelection;
import swjtu.ml.filter.supervised.FARNeM;
import swjtu.ml.filter.supervised.RSFSAID;
import swjtu.ml.filter.supervised.WAR;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

import java.awt.*;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by xin on 2017/4/6.
 */
public class WekaTest {
    ThresholdVisualizePanel tvp = new ThresholdVisualizePanel();

    private ArrayList<String> resultStrings = new ArrayList<String>();

    public static void main(String[] args) throws Exception{
        File file = new File("/Users/xin/Desktop/ExperimentData/binaryARFF/bands.arff");
        Instances instances = new Instances(new FileReader(file));
        instances.setClassIndex(instances.numAttributes()-1);

        RSFSAID rsfsaid = new RSFSAID(0.064,0.44,0);

        FARNeM farNeM = new FARNeM(0.125);
        WAR war = new WAR(0.1);

        FeatureSelection fs = new FeatureSelection(rsfsaid);
        FeatureSelection fs1 = new FeatureSelection(war);
        FeatureSelection fs2 = new FeatureSelection(farNeM);

        fs.setInputFormat(instances);
        fs1.setInputFormat(instances);
        fs2.setInputFormat(instances);
        WekaTest wt = new WekaTest();

//        Discretize discretize = new Discretize();
//        discretize.setInputFormat(instances);
//        instances = Filter.useFilter(instances,discretize);
        try {
//            String r = fs.selectFeature(instances);
//            System.out.println(r);
            Instances newInstances = Filter.useFilter(instances,fs);
            wt.addPlot(new J48(), newInstances,"RSFSAID");



//            r = fs1.selectFeature(instances);
//            System.out.println(r);
            Instances newInstances1 = Filter.useFilter(instances,fs1);
            wt.addPlot(new J48(), newInstances1,"WAR");

//            r = fs2.selectFeature(instances);
//            System.out.println(r);
            Instances newInstances2 = Filter.useFilter(instances,fs2);
            wt.addPlot(new J48(), newInstances2,"FARNeM");

        }catch (FSException e){
            e.printStackTrace();
        }

        wt.addPlot(new J48(), instances, "original");
        wt.plot("original");

    }

    public void plot(String title) throws Exception {
        String plotName = tvp.getName();
        final javax.swing.JFrame jf =
                new javax.swing.JFrame(title+": "+plotName);
        jf.setSize(500,400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(tvp,BorderLayout.CENTER);
        jf.addWindowListener(new java.awt.event.WindowAdapter(){
            @Override
            public void windowClosing(WindowEvent e) {
                jf.dispose();
            }
        });
        jf.setVisible(true);
    }

    public void addPlot(Classifier classifier, Instances instances, String plotName) throws Exception {
        Evaluation eval = new Evaluation(instances);
        eval.crossValidateModel(classifier,instances,5,new Random(1));
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 0;
        Instances result = tc.getCurve(eval.predictions());
//        tvp.setROCString("(AUC = "+
//                Utils.doubleToString(tc.getROCArea(result),4)+")");
        tvp.setName(result.relationName());
        PlotData2D pd = new PlotData2D(result);
        pd.setPlotName(plotName+"-"+Utils.doubleToString(tc.getROCArea(result),4));
        pd.addInstanceNumberAttribute();
        boolean[] cp = new boolean[result.numInstances()];
        for (int i=1;i<cp.length;i++) cp[i]=true;
        pd.setConnectPoints(cp);
        tvp.addPlot(pd);
    }


    public void output() throws Exception {
        File file = new File("/Users/xin/workspase/DataSet/ionosphere/result.csv");
        if(!file.getParentFile().exists()){
            file.getParentFile().mkdirs();
        }
        if(!file.exists()){
            file.createNewFile();
        }
        PrintWriter pw = new PrintWriter(file);

    }
}
