package pers.xin.test;

import swjtu.ml.filter.FeatureSelection;
import swjtu.ml.filter.supervised.FARNeM;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;

import java.io.File;
import java.io.FileReader;
import java.util.Random;

/**
 * Created by xin on 2017/5/18.
 */
public class AlgorithmTest {
    public static void main(String[] args) throws Exception {
        File file = new File("/Users/xin/Desktop/ExperimentData/myDataARFF/ionosphere.arff");
        Instances instances = new Instances(new FileReader(file));
        instances.setClassIndex(instances.numAttributes() - 1);

        FARNeM farNeM = new FARNeM(0.125);
        FeatureSelection fs = new FeatureSelection(farNeM);
        fs.setInputFormat(instances);
        String result = fs.selectFeature(instances);
        System.out.println(result);
        Instances newIns = Filter.useFilter(instances,fs);
        Evaluation eval = new Evaluation(newIns);
        eval.crossValidateModel(new weka.classifiers.functions.LibSVM(),instances,10,new Random(1));
        System.out.println(eval.pctCorrect());
    }
}
