package swjtu.ml.filter.supervised;

import swjtu.ml.utils.MyEuclideanDistance;
import weka.core.Instances;

import java.util.HashSet;

/**
 * Created by xin on 2017/5/9.
 * [1] J. Liu, Q. Hu, and D. Yu, “A weighted rough set based method developed for class imbalance learning,” Inf. Sci. (Ny)., vol. 178, no. 4, pp. 1235–1256, 2008.
 */
public class WAR {
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

}
