package pers.xin.optimization;

/**
 * Created by xin on 2017/4/27.
 */
public class AUCFSFitness extends BaseFitness{
    private int featureCount;

    public AUCFSFitness(double AUC, int featureCount){
        super(AUC);
        this.featureCount=featureCount;
    }

    public AUCFSFitness(double AUC){
        super(AUC);
        //this.featureCount=featureCount;
    }

    public boolean isBetterThan(Fitness fitness) {
        if(fitness==null) return true;
        AUCFSFitness f = (AUCFSFitness) fitness;
        //if(this.m_fitness <f.m_fitness ||(this.m_fitness ==f.m_fitness &&this.featureCount>f.featureCount))
        // return false;
        if(this.m_fitness <f.m_fitness)return false;
        return true;
    }

    public AUCFSFitness copy(){
        return new AUCFSFitness(this.m_fitness,this.featureCount);
    }
}
