package pers.xin.optimization;

/**
 * Created by xin on 2017/4/19.
 */
public interface Optimizable {

    /**
     * 计算适应度
     * @param params
     * @return
     */
    double fitness(double[] params);

    /**
     * 获得各个参数对应的区间；
     * 如第一个参数取值区间： getInterval[0][0]-getInterval[0][1]
     * @return
     */
    double[][] getInterval();
}
