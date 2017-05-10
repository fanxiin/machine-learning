package pers.xin.Experiment;

import pers.xin.optimization.AUCFSFitness;
import pers.xin.optimization.Fitness;
import pers.xin.optimization.Optimizable;
import swjtu.ml.filter.FSAlgorithm;
import swjtu.ml.filter.FSException;
import swjtu.ml.filter.FeatureSelection;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

import java.util.HashMap;
import java.util.Random;


/**
 * Created by xin on 2017/4/19.
 */
public class Experiment extends Optimizable{
    private Instances data;
    private int positiveIndex;
    private String classifierName;
    private String FSAlgorithmName;
    private HashMap<String,Double> featuresAUC=new HashMap<String, Double>();
    private HashMap<String,Double> paramsAUCBuffer = new HashMap<String, Double>();
    private HashMap<String,Integer> paramsFeatureCountBuffer = new HashMap<String, Integer>();
    private FormatSummary summary;

    public Experiment(String classifierName, Instances data, FormatSummary summary){
        this.classifierName =classifierName;
        this.data=data;
        this.summary = summary;
        int[] counts = new int[2];
        for (Instance datum : data) {
            counts[(int)datum.classValue()]++;
        }
        positiveIndex = counts[0]<counts[1]?0:1;
    }

    public void setFSAlgorithmName(String FSAlgorithmName) {
        this.FSAlgorithmName = FSAlgorithmName;
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
            return FSTest(params);
        } catch (Exception e) {
            if(e instanceof FSException) System.out.println(e.getMessage());
            return new AUCFSFitness(0,0);
        }
    }


    /**
     * 使用FS算法特征选择测试分类
     * @param params
     * @return 返回AUC值
     * @throws Exception 参数不合适时抛出FSException
     */
    public AUCFSFitness FSTest(double[] params) throws Exception {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < params.length; i++) {
            sb.append(params[i]+",");
        }
        sb.deleteCharAt(sb.length()-1);
        String paramsCode = sb.toString();

        if(!paramsAUCBuffer.containsKey(paramsCode)){
            FSAlgorithm algorithm = (FSAlgorithm) Class.forName(FSAlgorithmName).newInstance();
            algorithm.setParams(params);
            FeatureSelection fs = new FeatureSelection(algorithm);
            fs.setInputFormat(data);
            String features = fs.selectFeature(data);
            paramsFeatureCountBuffer.put(paramsCode,algorithm.getSelectedAttributes().length);
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
    public String originalAnalyze() throws Exception {
        Evaluation eval = new Evaluation(data);
        Classifier classifier = (Classifier) Class.forName(classifierName).newInstance();
        eval.crossValidateModel(classifier,data,10,new Random(1));
        return summary.format(eval,positiveIndex);
    }

    /**
     * 获得分类器对算法特征选择后的训练集的数据训练得出的分类指标
     * @param params
     * @return
     * @throws Exception
     */
    public String FSAnalyze(double[] params) throws Exception {
        FSAlgorithm algorithm = (FSAlgorithm) Class.forName(FSAlgorithmName).newInstance();
        algorithm.setParams(params);
        FeatureSelection fs = new FeatureSelection(algorithm);
        fs.setInputFormat(data);
        String reduction = fs.selectFeature(data);
        Instances fs_data = Filter.useFilter(data,fs);
        Evaluation eval = new Evaluation(fs_data);
        Classifier classifier = (Classifier) Class.forName(classifierName).newInstance();
        eval.crossValidateModel(classifier,fs_data,10,new Random(1));
        return summary.format(params, reduction, eval, positiveIndex);
    }

    /**
     * 对训练集进行特征选择
     * @param params
     * @return 选择后的数据
     * @throws Exception 参数不合适时抛出FSException
     */
    public Instances trainSetFS(double[] params) throws Exception {
        FSAlgorithm algorithm = (FSAlgorithm) Class.forName(FSAlgorithmName).newInstance();
        algorithm.setParams(params);
        FeatureSelection fs = new FeatureSelection(algorithm);
        fs.setInputFormat(data);
        return Filter.useFilter(data,fs);
    }
}


