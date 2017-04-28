package pers.xin.optimization;

/**
 * Created by xin on 2017/4/19.
 */
public abstract class Optimizable {
    protected double[][] interval;
    protected int[] precision;

    /**
     * 计算适应度
     * @param params
     * @return
     */
    public abstract Fitness computeFitness(double[] params);

    public double[][] getInterval() {
        return interval;
    }

    /**
     * 获得各个参数对应的区间；
     * 如第一个参数取值区间： getInterval[0][0]-getInterval[0][1]
     * @return
     */
    public void setInterval(double[][] interval) {
        this.interval = interval;
    }

    public int[] getPrecision() {
        return precision;
    }

    public void setPrecision(int[] precision) {
        this.precision = precision;
    }
}
