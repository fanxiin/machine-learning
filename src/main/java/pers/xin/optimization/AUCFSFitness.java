package pers.xin.optimization;

/**
 * Created by xin on 2017/4/27.
 */
public class AUCFSFitness implements Fitness{
    private double AUC;
    private int featureCount;

    public AUCFSFitness(double AUC, int featureCount){
        this.AUC=AUC;
        this.featureCount=featureCount;
    }

    public boolean isBetterThan(Fitness fitness) {
        if(fitness==null) return true;
        AUCFSFitness f = (AUCFSFitness) fitness;
        if(this.AUC<f.AUC||(this.AUC==f.AUC&&this.featureCount>f.featureCount))return false;
        return true;
    }
}
