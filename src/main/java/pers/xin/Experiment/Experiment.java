package pers.xin.Experiment;

import pers.xin.optimization.Optimizable;
import swjtu.ml.filter.FSException;
import swjtu.ml.filter.FeatureSelection;
import swjtu.ml.filter.supervised.RSFSAID;
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
public class Experiment implements Optimizable{
    private Instances data;
    private int positiveIndex;
    private String classifierName;
    private double[][] interval;
    private HashMap<String,Double> featuresAUC=new HashMap<String, Double>();
    private HashMap<Long,Double> AUCBuffer = new HashMap<Long, Double>();

    public Experiment(String classifierName, Instances data){
        this.classifierName =classifierName;
        this.data=data;
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
    public double fitness(double[] params){
        double AUC = 0;
//        for (int i = 0; i < params.length; i++) {
//            BigDecimal b = new BigDecimal(params[i]);
//            params[i] = b.setScale(3,BigDecimal.ROUND_HALF_UP).doubleValue();
//        }
        try {
            AUC = RSFSAIDTest(params[0],params[1],params[2]);
        } catch (FSException e) {
            System.out.println(e.getMessage());
            return 0;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return -AUC;
    }

    /**
     * 设置参数区间
     * @param interval
     */
    public void setInterval(double[][] interval) {
        this.interval = interval;
    }

    public double[][] getInterval() {
        return interval;
    }

    /**
     * 使用RSFSAID算法特征选择测试分类
     * @param delta
     * @param alpha
     * @param beta
     * @return 返回AUC值
     * @throws Exception 参数不合适时抛出FSException
     */
    public double RSFSAIDTest(double delta, double alpha, double beta) throws Exception {
        long paramsCode = Math.round(delta*100)*10000+Math.round(alpha*100)*100+Math.round(beta*100);
        if(!AUCBuffer.containsKey(paramsCode)){
            RSFSAID rsfsaid = new RSFSAID(delta,alpha,beta);
            FeatureSelection fs = new FeatureSelection(rsfsaid);
            fs.setInputFormat(data);
            String features = fs.selectFeature(data);
            if(!featuresAUC.containsKey(features)){
                Instances fs_data = Filter.useFilter(data,fs);
                Classifier classifier = (Classifier) Class.forName(classifierName).newInstance();
                featuresAUC.put(features,computeAUC(fs_data,classifier));
            }
            AUCBuffer.put(paramsCode,featuresAUC.get(features));
        }
        return AUCBuffer.get(paramsCode);
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

