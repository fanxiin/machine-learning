package swjtu.ml.filter.supervised;

import swjtu.ml.filter.FSAlgorithm;

import swjtu.ml.utils.MyEuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.util.*;

/**
 * Created by xin on 2017/5/9.
 * 只能处理离散型数据
 *
 * [1] J. Liu, Q. Hu, and D. Yu, “A weighted rough set based method developed for class imbalance learning,” Inf. Sci. (Ny)., vol. 178, no. 4, pp. 1235–1256, 2008.
 */
public class WAR implements FSAlgorithm{
    /**
     * 训练数据
     */
    private Instances m_data;

    /**
     * 保存约简过程中选择出来的属性
     */
    private HashSet<Integer> tempReduction;
    /**
     * 保存当前选择出来的属性的依赖度，对应red
     */
    private double dependency;

    private double sumWeight;

    private double sigma;

    /**
     * holds the selected attributes
     */
    private int[] m_SelectedAttributes;

    private String s_SelectedAttributes = "";

    private MyEuclideanDistance m_EuclideanDistance;

    HashMap<String, Double> classWeight;

    public WAR(double sigma, HashMap<String,Double> classWeight){
        this.sigma=sigma;
        this.classWeight=classWeight;
    }

    public WAR(){}

    public int[] SelectAttributes(Instances data) throws Exception {

        /**对数据进行离散化*/
        Discretize discretize = new Discretize();
        discretize.setInputFormat(data);
        data = Filter.useFilter(data,discretize);
        initFeatureSelection(data);

        /**设置权重*/
        if(classWeight!=null)
            setWeight();

        int numAttr = m_data.numAttributes()-1;
        /**计算H(D|C)*/
        HashSet<Integer> allAttrs = new HashSet<Integer>();
        for (int i = 0; i < numAttr; i++) {
            allAttrs.add(i);
        }
        double allAttrsConditionEntropy = weightedConditionEntropy(findEquivalenceSets(allAttrs));
        for (int i = 0; i < numAttr; i++) {
            double maxSGF = 0;
            int bestAttr = -1;
            for (int j = 0; j < numAttr; j++) {
                if(!tempReduction.contains(j)){
                    double sgf = computeSGF(j);
                    if(sgf>maxSGF){
                        maxSGF = sgf;
                        bestAttr = j;
                    }
                }
            }
            if(bestAttr>=0){
                tempReduction.add(bestAttr);
                dependency = weightedDependency(tempReduction);
                if(weightedConditionEntropy(findEquivalenceSets(tempReduction))-allAttrsConditionEntropy<sigma){
                    break;
                }
            }
        }


        HashSet<Integer> reduction = new HashSet<Integer>();
        reduction.addAll(tempReduction);
        for (Integer attr : tempReduction) {
            reduction.remove(attr);
            if(weightedConditionEntropy(findEquivalenceSets(reduction))-allAttrsConditionEntropy>=sigma){
                reduction.add(attr);
            }
        }
        ArrayList<Integer> selectedAttributes = new ArrayList<Integer>();
        selectedAttributes.addAll(reduction);
        selectedAttributes.add(m_data.classIndex());
        Collections.sort(selectedAttributes);
        m_SelectedAttributes = new int[selectedAttributes.size()];
//        System.out.println(selectedAttributes.toString());
        for (int col = 0; col < selectedAttributes.size(); col++) {
            m_SelectedAttributes[col] = selectedAttributes.get(col);
        }
        s_SelectedAttributes = selectedAttributes.toString();
        return m_SelectedAttributes;
    }

    public String getSelectedAttributesString() {
        return this.s_SelectedAttributes;
    }

    public int[] getSelectedAttributes() {
        return this.m_SelectedAttributes;
    }

    public void setParams(double[] params) {
        this.sigma = params[0];
    }

    public void setWeight(HashMap<String, Double> classWeight) {
        this.classWeight = classWeight;
    }

    /**
     * 加权分类质量
     * @param attrIndex
     * @return
     */
    private double computeSGF(int attrIndex) throws Exception {
        HashSet<Integer> redAndAttr = new HashSet<Integer>();
        redAndAttr.addAll(tempReduction);
        redAndAttr.add(attrIndex);
        return weightedDependency(redAndAttr) - dependency;
    }

    private double weightedDependency(HashSet<Integer> attrIndexes) throws Exception {
        HashSet<HashSet<Integer>> equivalenceSets = findEquivalenceSets(attrIndexes);
        double posWeight = 0;
        for(HashSet<Integer> equivalenceClass : equivalenceSets){
            boolean isPos = true;
            Iterator<Integer> i = equivalenceClass.iterator();
            double firstInstanceClass = m_data.get(i.next()).classValue();
            while (i.hasNext()){
                if(firstInstanceClass!=m_data.get(i.next()).classValue()){
                    isPos = false;
                    break;
                }
            }
            if(isPos){
                for (Integer instanceIndex : equivalenceClass) {
                    posWeight += m_data.get(instanceIndex).weight();
                }
            }
        }
        return posWeight/sumWeight;
    }

    /**
     * 计算领域集合
     */
    private HashSet<HashSet<Integer>> findEquivalenceSets(HashSet<Integer> attrSet) throws Exception {
        /** 转换需要计算的属性所在列（从1开始） */
        ArrayList<Integer> attrArray = new ArrayList<Integer>();
        attrArray.addAll(attrSet);
        for (int i = 0; i < attrArray.size(); i++) {
            attrArray.set(i,attrArray.get(i)+1);
        }
        Collections.sort(attrArray);
        StringBuilder attrString = new StringBuilder();
        for (int index : attrArray) attrString.append(index + ",");
        attrString.deleteCharAt(attrString.length()-1);
        m_EuclideanDistance.setAttributeIndices(attrString.toString());
        int dataCount = m_data.numInstances();
        int[] visitedFlag = new int[dataCount];

        HashSet<HashSet<Integer>> equivalenceSets = new HashSet<HashSet<Integer>>();
        for (int i = 0; i < dataCount; i++) {
            if(visitedFlag[i]==0){
                HashSet<Integer> equivalenceClass = new HashSet<Integer>();
                for (int j = i; j < dataCount; j++) {
                    double m_distance = m_EuclideanDistance.distance(m_data.get(i),
                            m_data.get(j));
                    if (m_distance == 0) {
                        equivalenceClass.add(j);
                        visitedFlag[j] = 1;
                    }
                }
                equivalenceSets.add(equivalenceClass);
            }
        }
        return equivalenceSets;
    }

    /**
     * 初始化分类器
     *
     * @param data
     * @throws Exception
     */
    private void initFeatureSelection(Instances data) throws Exception {
        m_data = data;
        m_EuclideanDistance = new MyEuclideanDistance(data);
        tempReduction = new HashSet<Integer>();
        double sw = 0.0;
        for (Instance m_datum : m_data) {
            sw += m_datum.weight();
        }
        sumWeight = sw;
    }
    /**
     * 计算加权熵H(B)
     * @param equivalenceSets B 所决定的等价类集合
     * @return
     * @throws Exception
     */
    private double weightedEntropy(HashSet<HashSet<Integer>> equivalenceSets) throws Exception {
        int instanceCount = m_data.numInstances();
        double entropy = 0;
        for (HashSet<Integer> equivalenceSet : equivalenceSets) {
            double weight = 0;
            for (Integer instanceIndex : equivalenceSet) {
                weight += m_data.get(instanceIndex).weight();
            }
            double probability = equivalenceSet.size()*1.0/instanceCount;
            entropy -= weight*probability*(Math.log(probability)/Math.log(2));
        }
        return entropy;
    }

    /**
     * 计算加权条件熵H(D|B)
     * @param equivalenceSets B 所决定的等价类集合
     * @return
     */
    private double weightedConditionEntropy(HashSet<HashSet<Integer>> equivalenceSets){
        double entropy = 0;
        for (HashSet<Integer> equivalenceSet : equivalenceSets) {
            double[] weightDi = new double[m_data.numClasses()];
            int[] counts = new int[m_data.numClasses()];
            for (Integer instanceIndex : equivalenceSet) {
                Instance ins = m_data.get(instanceIndex);
                weightDi[(int)ins.classValue()]+=ins.weight();
                counts[(int)ins.classValue()]++;
            }
            for (int i = 0; i < weightDi.length; i++) {
                double probability = counts[i]*1.0/equivalenceSet.size();
                entropy -= weightDi[i]*probability*log2(probability);
            }
        }
        return entropy;
    }

    private double log2(double x){
        if(x==0)return 0;
        return Math.log(x)/Math.log(2);
    }

    private void setWeight(){
        for (Map.Entry<String, Double> stringDoubleEntry : classWeight.entrySet()) {
            double valueIndex = m_data.classAttribute().indexOfValue(stringDoubleEntry.getKey());
            for (Instance m_datum : m_data) {
                if(valueIndex == m_datum.classValue())
                    m_datum.setWeight(stringDoubleEntry.getValue());
            }
        }
    }

}
