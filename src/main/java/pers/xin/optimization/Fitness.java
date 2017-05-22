package pers.xin.optimization;

/**
 * Fitness
 * Created by xin on 2017/4/27.
 */
public class Fitness {

    protected double m_fitness;

    public Fitness(double fitness){
        this.m_fitness=fitness;
    }

    /**
     * 若参数为null必须返回true;
     * @param fitness
     * @return
     */
    boolean isBetterThan(Fitness fitness){
        if(fitness==null) return true;
        BaseFitness bf = (BaseFitness) fitness;
        return m_fitness<bf.m_fitness?true:false;
    }

    public double fitness() {
        return m_fitness;
    }
}
