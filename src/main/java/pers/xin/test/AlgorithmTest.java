package pers.xin.test;

import swjtu.ml.filter.FeatureSelection;
import swjtu.ml.filter.supervised.*;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;

import java.io.File;
import java.io.FileReader;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by xin on 2017/5/18.
 */
public class AlgorithmTest {
    public static void main(String[] args) throws Exception {
        File file = new File("/Users/xin/Desktop/ExperimentData/test/ionosphere.arff");
        Instances instances = new Instances(new FileReader(file));
        instances.setClassIndex(instances.numAttributes() - 1);

        RSFSAID2 rsfsaid = new RSFSAID2(0.37,0,0.52);
        FeatureSelection fs = new FeatureSelection(rsfsaid);
        fs.setInputFormat(instances);
        int[] result = fs.selectFeature(instances);
        System.out.println(Arrays.toString(result));
        Instances newIns = Filter.useFilter(instances,fs);
        Evaluation eval = new Evaluation(newIns);
        eval.crossValidateModel(new weka.classifiers.trees.J48(),newIns,5,new Random(1));
        System.out.println(eval.areaUnderROC(0));
    }
}
