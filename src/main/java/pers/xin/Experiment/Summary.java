package pers.xin.Experiment;

import weka.classifiers.Evaluation;

/**
 * Created by xin on 2017/6/12.
 */
public class Summary {
    private Evaluation eval;
    private int positiveIndex;
    private double[] params;
    private int[] reduction;

    public Summary(Evaluation eval,int positiveIndex){
        this.eval = eval;
        this.positiveIndex = positiveIndex;
    }

    public Summary(Evaluation eval,int positiveIndex, double[] params, int[] reduction){
        this(eval,positiveIndex);
        this.params = params;
        this.reduction = reduction;
    }

    public Evaluation getEval() {
        return eval;
    }

    public double[] getParams() {
        return params;
    }

    public int[] getReduction() {
        return reduction;
    }

    public int getPositiveIndex() {
        return positiveIndex;
    }

    public boolean isReducted() {
        return reduction == null ? false : true;
    }
}
