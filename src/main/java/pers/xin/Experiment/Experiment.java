package pers.xin.Experiment;

import pers.xin.optimization.AUCFSFitness;
import pers.xin.optimization.Fitness;
import pers.xin.optimization.Optimizable;
import swjtu.ml.filter.FSException;
import swjtu.ml.filter.FeatureSelection;
import swjtu.ml.filter.supervised.RSFSAID;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

import java.text.NumberFormat;
import java.util.HashMap;
import java.util.Random;


/**
 * Created by xin on 2017/4/19.
 */
public class Experiment extends Optimizable{
    private Instances data;
    private int positiveIndex;
    private String classifierName;
    private HashMap<String,Double> featuresAUC=new HashMap<String, Double>();
    private HashMap<String,Double> paramsAUCBuffer = new HashMap<String, Double>();
    private HashMap<String,Integer> paramsFeatureCountBuffer = new HashMap<String, Integer>();
    private NumberFormat numberFormat = NumberFormat.getNumberInstance();

    public Experiment(String classifierName, Instances data){
        this.classifierName =classifierName;
        this.data=data;
        numberFormat.setMaximumFractionDigits(2);
        int[] counts = new int[2];
        for (Instance datum : data) {
            counts[(int)datum.classValue()]++;
        }
        positiveIndex = counts[0]<counts[1]?0:1;
    }

    public String getClassifierName() {
        return classifierName;
    }

    /**
     * 计算适应度
     * @param params
     * @return
     */
    public Fitness computeFitness(double[] params){
        //Fitness AUC = new AUCFSFitness(0,0);
//        for (int i = 0; i < params.length; i++) {
//            BigDecimal b = new BigDecimal(params[i]);
//            params[i] = b.setScale(3,BigDecimal.ROUND_HALF_UP).doubleValue();
//        }
        try {
            return RSFSAIDTest(params[0],params[1],params[2]);
        } catch (Exception e) {
            if(e instanceof FSException) System.out.println(e.getMessage());
            return new AUCFSFitness(0,0);
        }
    }


    /**
     * 使用RSFSAID算法特征选择测试分类
     * @param delta
     * @param alpha
     * @param beta
     * @return 返回AUC值
     * @throws Exception 参数不合适时抛出FSException
     */
    public AUCFSFitness RSFSAIDTest(double delta, double alpha, double beta) throws Exception {
//        String paramsCode = numberFormat.format(delta)+","+numberFormat.format(alpha)+","+numberFormat.format(beta);
        String paramsCode = delta+","+alpha+","+beta;

        if(!paramsAUCBuffer.containsKey(paramsCode)){
            RSFSAID rsfsaid = new RSFSAID(delta,alpha,beta);
            FeatureSelection fs = new FeatureSelection(rsfsaid);
            fs.setInputFormat(data);
            String features = fs.selectFeature(data);
            paramsFeatureCountBuffer.put(paramsCode,rsfsaid.getSelectedAttributeCount());
            if(!featuresAUC.containsKey(features)){
                Instances fs_data = Filter.useFilter(data,fs);
                Classifier classifier = (Classifier) Class.forName(classifierName).newInstance();
                featuresAUC.put(features,computeAUC(fs_data,classifier));
            }
            paramsAUCBuffer.put(paramsCode,featuresAUC.get(features));
        }
        double AUC = paramsAUCBuffer.get(paramsCode);
        int featureCount = paramsFeatureCountBuffer.get(paramsCode);
        return new AUCFSFitness(AUC,featureCount);
    }

    /**
     * 使用未特征选择的测试分类
     * @return
     * @throws Exception
     */
    public double originalTest() throws Exception {
        Classifier classifier = (Classifier) Class.forName(classifierName).newInstance();
        return computeAUC(data,classifier);
    }

    /**
     * 计算分类的AUC指标
     * @param data
     * @param classifier
     * @return
     * @throws Exception
     */
    public double computeAUC(Instances data,Classifier classifier) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier,data,10,new Random(1));
        return eval.areaUnderROC(positiveIndex);
    }

    /**
     * 获得分类器对原始训练数据的分类指标
     * @return
     * @throws Exception
     */
    public Summary originalAnalyze() throws Exception {
        Evaluation eval = new Evaluation(data);
        Classifier classifier = (Classifier) Class.forName(classifierName).newInstance();
        eval.crossValidateModel(classifier,data,10,new Random(1));
        return new Summary(eval,positiveIndex);
    }

    /**
     * 获得分类器对RSFSAID算法特征选择后的训练集的数据训练得出的分类指标
     * @param params
     * @return
     * @throws Exception
     */
    public Summary RSFSAIDAnalyze(double[] params) throws Exception {
//        for (int i = 0; i < params.length; i++) {
//            BigDecimal bg = new BigDecimal(params[i]);
//            params[i] = bg.setScale(2,BigDecimal.ROUND_HALF_UP).doubleValue();
//        }
        RSFSAID rsfsaid = new RSFSAID(params[0],params[1],params[2]);
        FeatureSelection fs = new FeatureSelection(rsfsaid);
        fs.setInputFormat(data);
        String reduction = fs.selectFeature(data);
        Instances fs_data = Filter.useFilter(data,fs);
        Evaluation eval = new Evaluation(fs_data);
        Classifier classifier = (Classifier) Class.forName(classifierName).newInstance();
        eval.crossValidateModel(classifier,fs_data,10,new Random(1));
        return new Summary(params, reduction, eval, positiveIndex);
    }

    /**
     * 对训练集进行特征选择
     * @param delta
     * @param alpha
     * @param beta
     * @return 选择后的数据
     * @throws Exception 参数不合适时抛出FSException
     */
    public Instances trainSetFS(double delta, double alpha, double beta) throws Exception {
        RSFSAID rsfsaid = new RSFSAID(delta,alpha,beta);
        FeatureSelection fs = new FeatureSelection(rsfsaid);
        fs.setInputFormat(data);
        return Filter.useFilter(data,fs);
    }
}


