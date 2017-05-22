package pers.xin.optimization;

/**
 * Created by xin on 2017/4/27.
 */
public class BaseFitness extends Fitness {
    double m_fitness;
    public BaseFitness(double num){
          super(num);
    }
    public boolean isBetterThan(Fitness fitness) {
        if(fitness==null) return true;
        BaseFitness bf = (BaseFitness) fitness;
        return m_fitness<bf.m_fitness?true:false;
    }
}
