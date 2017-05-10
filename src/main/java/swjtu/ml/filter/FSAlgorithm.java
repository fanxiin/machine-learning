package swjtu.ml.filter;

import weka.core.Instances;

/**
 * Created by xin on 2017/4/13.
 */
public interface FSAlgorithm {
    /**
     * 实施特征选择，选出子集的索引存储在m_SelectedAttributes（其中包含class）中。
     * @param data
     */
    int[] SelectAttributes(Instances data) throws Exception;

    String getSelectedAttributesString();

    int[] getSelectedAttributes();

    void setParams(double[] params);
}
