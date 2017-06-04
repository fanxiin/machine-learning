package swjtu.ml.filter.supervised;

import swjtu.ml.filter.FSAlgorithm;
import swjtu.ml.filter.FSException;
import swjtu.ml.utils.MyEuclideanDistance;
import swjtu.ml.utils.Tuple2;
import weka.core.*;

import java.util.*;

/**
 * Created by xin on 2017/4/13.
 */
public class RSFSAID implements FSAlgorithm {
    /**
     * 领域粗糙集的距离阈值
     */
    private double delta;

    /**
     * 属性重要度参数
     */
    private double alpha, beta;

    /**
     * 训练数据
     */
    private Instances m_data;

    private int numNumrice;

    /**
     * 领域集合
     */
    private int[][] neighborSets;

    /**
     * 领域集合对应的泛化决策;
     * -1: 正负类都可能;
     * 其他对应正负类索引
     */
    private int[] generalDecision;

    /** 索引对应的对象的类标（按weka内部的方式将double转为int） */
    private int[] instanceClass;

    /** 正负类对象个数 */
    private int[] classCount;

    /**
     * 正类对应的统计的索引（正类标值）
     */
    private int posIndex;

    /**
     * 负类对应的统计的索引（负类标值）
     */
    private int negIndex;

    /**
     * holds the selected attributes
     */
    private int[] m_SelectedAttributes;

    private String s_SelectedAttributes = "";

    private MyEuclideanDistance m_EuclideanDistance;

    /**
     * 特征选择算法构造函数
     *
     * @param delta
     * @param alpha
     * @param beta
     */
    public RSFSAID(double delta, double alpha, double beta) {
        this.delta = delta;
        this.alpha = alpha;
        this.beta = beta;
    }

    public RSFSAID(){}

    public int[] getSelectedAttributes() {
        return m_SelectedAttributes;
    }

    public void setParams(double[] params) {
        this.delta = params[0];
        this.alpha = params[1];
        this.beta = params[2];
    }

    public String getSelectedAttributesString() {
        return s_SelectedAttributes;
    }

    public int getSelectedAttributeCount(){
        return  m_SelectedAttributes.length;
    }

    /**
     * 计算领域集合,集领域对应的泛化决策
     */
    private void findNeighborhoodSets() throws Exception {
        int dataCount = m_data.numInstances();
        neighborSets = new int[dataCount][dataCount];
        double m_distance=0.0;
        for (int i = 0; i < dataCount; i++) {
            for (int j = i; j < dataCount; j++) {
                if(numNumrice!=0)
                    m_distance = m_EuclideanDistance.distance(m_data.get(i),
                        m_data.get(j)) / numNumrice;
                else
                    m_distance = m_EuclideanDistance.distance(m_data.get(i),
                            m_data.get(j));
                if (m_distance <= delta) {
                    neighborSets[i][j] = 1;
                    neighborSets[j][i] = 1;
                }
            }
        }

        /** 计算领域对应的泛化决策 */
        generalDecision = new int[dataCount];

        for (int i = 0; i < neighborSets.length; i++) {
            double oneClass = m_data.get(i).classValue();
            for (int j = 1; j < neighborSets.length; j++) {
                /** 确保j属于的领域 */
                if (neighborSets[i][j] == 1) {
                    /** 此领域内已发现同时包含正负类，泛化决策取值-1表示泛化决策为正负类 */
                    if (oneClass != m_data.get(j).classValue()) {
                        generalDecision[i] = -1;
                        break;
                    }
                    generalDecision[i] = (int) oneClass;
                }
            }
        }

    }


    private int[][] findNeighborhoodSetsByAttr(int attrIndex) throws Exception {
        int dataCount = m_data.numInstances();
        int[][] neighborSetsByAttr = new int[dataCount][dataCount];
        for (int i = 0; i < dataCount; i++) {
            for (int j = i; j < dataCount; j++) {
                double m_distance = m_EuclideanDistance.distanceOnAttr(attrIndex, m_data.get(i),
                        m_data.get(j));
                if (m_distance <= delta) {
                    neighborSetsByAttr[i][j] = 1;
                    neighborSetsByAttr[j][i] = 1;
                }
            }
        }
        return neighborSetsByAttr;
    }


    /**
     * 计算单个属性的重要度（属性列从第一列开始）
     *
     * @param attrIndex
     * @return
     */
    private double computeSignificance(int attrIndex) throws Exception {
        /** 按属性列获得领域集合 */
        int[][] m_neighborSets = findNeighborhoodSetsByAttr(attrIndex);

        int nPos = classCount[posIndex];
        int nNeg = classCount[negIndex];

        /** 可能的错正类（按领域内正负数据多数分）
         *  错分的上边界对象数
         *  上边界元素数大于下边界元素数的领域的所有下边界元素的个数和 */
        int FP = 0;
        /** 可能的错负类（按领域内正负数据多数分）
         *  错分的下边界对象
         *  下边界元素数大于上边界元素数的领域的所有上边界元素的个数和 */
        int FN = 0;

        for (int i = 0; i < m_neighborSets.length; i++) {
            /** 第一个为实际正类数，第二个位实际负类数 *//** 对应正负类索引 */
            int[] posAndNeg = new int[2];
            for (int j = 0; j < m_neighborSets.length; j++) {
                if (m_neighborSets[i][j] == 1) {
                    if (instanceClass[j] == posIndex)
                        posAndNeg[posIndex]++;
                    else
                        posAndNeg[negIndex]++;
                }
            }
            /**************** 如果该领域内正负类个数相等？ *******************/
            /** 下边界大于上边界，故给单个对象分类时将其分入正类，则领域中实际为负类的对象分错
             *  即错正类 */
            if (posAndNeg[posIndex] >= posAndNeg[negIndex] && instanceClass[i] == negIndex)
                FP++;
            /** 上边界大于下边界 */
            if (posAndNeg[negIndex] >= posAndNeg[posIndex] && instanceClass[i] == posIndex)
                FN++;
        }

        return 1 - (alpha * FP / nPos + beta * FN / nNeg) / 2;
    }


    /**
     * 并统计各类个数,并标记少数类为正类
     *
     * @throws Exception
     */
    private void computeClassSet() throws Exception {
        if (m_data.get(0).numClasses() != 2)
            throw new Exception("不是二分类数据！");
        classCount = new int[2];
        instanceClass = new int[m_data.numInstances()];
        for (int i = 0; i < m_data.numInstances(); i++) {
            int classVlaue = (int) m_data.get(i).classValue();
            instanceClass[i] = classVlaue;
            classCount[classVlaue]++;
        }
        /** 当两类对象数相等，令posIndex=0，negIndex=1 */
        posIndex = classCount[0] <= classCount[1] ? 0 : 1;

        negIndex = classCount[0] > classCount[1] ? 0 : 1;
    }

    private boolean discernible(int attrIndex, int i, int j) {
        return m_EuclideanDistance.distanceOnAttr(attrIndex, m_data.get(i), m_data.get(j)) > delta;
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
        numNumrice = 0;
        for (int i = 0; i < m_data.numAttributes() - 1; i++) {
            if (m_data.attribute(i).type() == Attribute.NUMERIC) {
                numNumrice++;
            }
        }
        computeClassSet();
        findNeighborhoodSets();
    }


    /**
     * 实施特征选择，选出子集的索引存储在m_SelectedAttributes（其中包含class）中。
     *
     * @param data
     */
    public int[] SelectAttributes(Instances data) throws Exception {
        initFeatureSelection(data);

        /** 重要度，_1:属性所对应列，_2:属性对应重要度 */
        ArrayList<Tuple2<Integer, Double>> significances = new ArrayList<Tuple2<Integer, Double>>();
        for (int k = 0; k < m_data.numAttributes() - 1; k++) {
            significances.add(new Tuple2<Integer, Double>(k, computeSignificance(k)));
        }

        Collections.sort(significances, new Comparator<Tuple2<Integer, Double>>() {
            public int compare(Tuple2<Integer, Double> o1, Tuple2<Integer, Double> o2) {
                if (o1._2() - o2._2() > 0)
                    return -1;
                else if (o1._2() - o2._2() < 0)
                    return 1;
                else
                    return 0;
            }
        });

        /** 计算可区分属性 */
        HashSet<HashSet<Integer>> discernibilityMatrix = new HashSet<HashSet<Integer>>();
        for (int i = 0; i < m_data.numInstances(); i++) {
            for (int j = 0; j < m_data.numInstances(); j++) {
                if (neighborSets[i][j] == 0 && generalDecision[i] != generalDecision[j]) {
                    HashSet<Integer> discernibilityAttr = new HashSet<Integer>();
                    for (int k = 0; k < m_data.numAttributes() - 1; k++) {
                        if (discernible(k, i, j))
                            discernibilityAttr.add(k);
                    }
                    discernibilityMatrix.add(discernibilityAttr);
                }
            }
        }

        /** 删除空集 */
        discernibilityMatrix.remove(new HashSet<HashSet<Integer>>());
        ArrayList<Integer> selectedAttributes = new ArrayList<Integer>();

        for (int i = 0; !discernibilityMatrix.isEmpty(); i++) {
            HashSet<HashSet<Integer>> tmp = new HashSet<HashSet<Integer>>();
            selectedAttributes.add(significances.get(i)._1());
            for (HashSet set : discernibilityMatrix) {
                if (set.contains(significances.get(i)._1()))
                    tmp.add(set);
            }
            discernibilityMatrix.removeAll(tmp);
        }

        if (selectedAttributes.isEmpty()) {
            throw new FSException("当前参数设置不当!" + "\ndelat:" + delta + ", alpha:" + alpha + ", beta:" + beta);
        }

        /** 加入类标类 */
        selectedAttributes.add(m_data.classIndex());
        Collections.sort(selectedAttributes, new Comparator<Integer>() {
            public int compare(Integer o1, Integer o2) {
                return o1 - o2;
            }
        });

        m_SelectedAttributes = new int[selectedAttributes.size()];
//        System.out.println(selectedAttributes.toString());
        for (int col = 0; col < selectedAttributes.size(); col++) {
            m_SelectedAttributes[col] = selectedAttributes.get(col);
        }
        s_SelectedAttributes = selectedAttributes.toString();
        return m_SelectedAttributes;
    }

}

