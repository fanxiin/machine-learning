package pers.xin.test;

import swjtu.ml.filter.FSException;
import swjtu.ml.filter.FeatureSelection;
import swjtu.ml.filter.supervised.RSFSAID;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
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
import java.util.Random;

/**
 * Created by xin on 2017/4/6.
 */
public class WekaTest {
    private ArrayList<String> resultStrings = new ArrayList<String>();

    public static void main(String[] args) throws Exception{
        File file = new File("/Users/xin/workspase/DataSet/download_keel/vehicle3.arff");
        Instances instances = new Instances(new FileReader(file));
        instances.setClassIndex(instances.numAttributes()-1);

        RSFSAID rsfsaid = new RSFSAID(0.03,0.7,0.3);

        FeatureSelection fs = new FeatureSelection(rsfsaid);

        Remove remove = new Remove();
        remove.setInputFormat(instances);
        String[] option = weka.core.Utils.splitOptions("-R 2,3,6");
        remove.setOptions(option);
        fs.setInputFormat(instances);
        WekaTest wt = new WekaTest();
        try {
            String r = fs.selectFeature(instances);
            System.out.println(r);
            Instances newInstances = Filter.useFilter(instances,fs);
            wt.plot(new J48(), newInstances,"FS");
        }catch (FSException e){
            e.printStackTrace();
        }
        wt.plot(new J48(),instances,"original");

    }

    public void plot(Classifier classifier, Instances instances,String title) throws Exception {

        Evaluation eval = new Evaluation(instances);
        eval.crossValidateModel(classifier,instances,10,new Random(1));

        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 0;
        Instances result = tc.getCurve(eval.predictions());

        ThresholdVisualizePanel tvp = new ThresholdVisualizePanel();
        tvp.setROCString("(AUC = "+
                Utils.doubleToString(tc.getROCArea(result),4)+")");
        tvp.setName(result.relationName());
        PlotData2D pd = new PlotData2D(result);
        pd.setPlotName(result.relationName());
        pd.addInstanceNumberAttribute();

        boolean[] cp = new boolean[result.numInstances()];
        for (int i=1;i<cp.length;i++) cp[i]=true;
        pd.setConnectPoints(cp);
        tvp.addPlot(pd);

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
