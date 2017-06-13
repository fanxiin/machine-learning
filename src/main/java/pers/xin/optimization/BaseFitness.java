package pers.xin.optimization;

/**
 * Created by xin on 2017/4/27.
 */
public class BaseFitness implements Fitness {
    protected double m_fitness;
    public BaseFitness(double num){
          this.m_fitness=num;
    }
    public boolean isBetterThan(Fitness fitness) {
        if(fitness==null) return true;
        BaseFitness bf = (BaseFitness) fitness;
        return m_fitness<bf.m_fitness?true:false;
    }

    public double fitness() {
        return m_fitness;
    }

    public BaseFitness copy(){
        return new BaseFitness(m_fitness);
    }
}
