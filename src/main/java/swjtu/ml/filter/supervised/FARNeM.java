package swjtu.ml.filter.supervised;

import swjtu.ml.filter.FSAlgorithm;
import swjtu.ml.utils.MyEuclideanDistance;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by xin on 2017/5/5.
 */
public class FARNeM implements FSAlgorithm {

    private double delta;

    /**
     * 训练数据
     */
    private Instances m_data;

    private int numNumrice;

    /**
     * 保存约简过程中选择出来的属性
     */
    private HashSet<Integer> red;
    /**
     * 保存当前选择出来的属性的依赖度，对应red
     */
    private double dependency;

    /**
     * holds the selected attributes
     */
    private int[] m_SelectedAttributes;

    private String s_SelectedAttributes = "";

    private MyEuclideanDistance m_EuclideanDistance;

    public FARNeM(double delta) {
        this.delta = delta;
    }

    public FARNeM(){}

    public int[] getSelectedAttributes() {
        return m_SelectedAttributes;
    }

    public void setParams(double[] params) {
        this.delta=params[0];
    }

    public void setWeight(HashMap<String, Double> classWeight) {
        throw new UnsupportedOperationException("暂不支持权重设置");
    }

    public int[] SelectAttributes(Instances data) throws Exception {
        initFeatureSelection(data);
        int numAttr = m_data.numAttributes()-1;
        double significance=0;
        do{
            significance = 0;
            int bestAttr=-1;
            for (int i = 0; i < numAttr; i++) {
                if(!red.contains(i)){
                    double tmpSig = computeSignificance(i);
                    if(tmpSig>significance){
                        significance=tmpSig;
                        bestAttr=i;
                    }
                }
            }
            if(significance>0){
                red.add(bestAttr);
                dependency = computeDependency(red);
            }

        }while(significance>0);
        System.out.println(dependency);
        ArrayList<Integer> selectedAttributes = new ArrayList<Integer>();
        selectedAttributes.addAll(red);
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
        return s_SelectedAttributes;
    }

    public int getSelectedAttributeCount() {
        return m_SelectedAttributes.length;
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
        red = new HashSet<Integer>();
        numNumrice = 0;
        for (int i = 0; i < m_data.numAttributes() - 1; i++) {
            if (m_data.attribute(i).type() == Attribute.NUMERIC) {
                numNumrice++;
            }
        }
    }

    private double computeDependency(HashSet<Integer> attrSet) throws Exception {
        int instanceCount = m_data.numInstances();
        int posCount = 0;
        int[][] neighborhoodSets = findNeighborhoodSets(attrSet);
        for (int i = 0; i < instanceCount; i++) {
            double classOfI = m_data.get(i).classValue();
            int j = 0;
            for (; j < instanceCount; j++) {
                if (neighborhoodSets[i][j] == 1 && m_data.get(j).classValue() != classOfI)
                    break;
            }
            if (j == instanceCount) posCount++;
        }
        return posCount*1.0 / instanceCount;
    }


    /**
     * 计算领域集合
     */
    private int[][] findNeighborhoodSets(HashSet<Integer> attrSet) throws Exception {
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
        int[][] neighborSets = new int[dataCount][dataCount];
        for (int i = 0; i < dataCount; i++) {
            for (int j = i; j < dataCount; j++) {
                double m_distance = m_EuclideanDistance.distance(m_data.get(i),
                        m_data.get(j))/numNumrice;
                if (m_distance <= delta) {
                    neighborSets[i][j] = 1;
                    neighborSets[j][i] = 1;
                }
            }
        }
        return neighborSets;
    }

    private double computeSignificance(int attrIndex) throws Exception {

        HashSet<Integer> redAndAttr = new HashSet<Integer>();
        redAndAttr.addAll(red);
        redAndAttr.add(attrIndex);
        return computeDependency(redAndAttr) - dependency;
    }



}
