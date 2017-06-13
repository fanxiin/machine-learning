package pers.xin.optimization;

/**
 * Fitness
 * Created by xin on 2017/4/27.
 */
public interface Fitness {
    /**
     * 若参数为null必须返回true;
     * @param fitness
     * @return
     */
    boolean isBetterThan(Fitness fitness);

    double fitness();

    Fitness copy();
}
