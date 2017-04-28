package pers.xin.Experiment;

import weka.classifiers.Evaluation;

/**
 * Created by xin on 2017/4/25.
 */
public class Summary {
    private double delta, alpha, beta;
    /**
     * Correctly Classified Instances Percentage
     */
    private double CCIP;
    /**
     * Incorrectly Classified Instances
     */
    private int Incorrectly_Classified_Instances;
    /**
     * Correctly_Classified_Instances
     */
    private int Correctly_Classified_Instances;
    /**
     * Incorrectly Classified Instances Percentage
     */
    private double ICIP;
    private double F_Measure, ROC_Area;
    private String reduction;
    private int reductionCount;
    private String dataSetName;
    private boolean isReduced;
    private int numAttribute;
    public static final String nullSymbol = "";

    public Summary(Evaluation eval, int positiveIndex) {
        this.Correctly_Classified_Instances = (int) eval.correct();
        this.Incorrectly_Classified_Instances = (int) eval.incorrect();
        this.CCIP = Correctly_Classified_Instances * 100.0 / (Correctly_Classified_Instances
                + Incorrectly_Classified_Instances);
        this.ICIP = Incorrectly_Classified_Instances * 100.0 / (Correctly_Classified_Instances
                + Incorrectly_Classified_Instances);
        this.F_Measure = eval.fMeasure(positiveIndex);
        this.ROC_Area = eval.areaUnderROC(positiveIndex);
        this.dataSetName = eval.getHeader().relationName();
        this.isReduced = false;
        this.numAttribute = eval.getHeader().numAttributes();
    }

    public Summary(double[] params, String reduction, Evaluation eval, int positiveIndex) {
        this(eval, positiveIndex);
        this.delta = params[0];
        this.alpha = params[1];
        this.beta = params[2];
        this.reduction = reduction;
        this.isReduced = true;
        this.reductionCount = reduction.split(",").length;
    }

    public double getDelta() {
        return delta;
    }

    public double getAlpha() {
        return alpha;
    }

    public double getBeta() {
        return beta;
    }

    public double getCCIP() {
        return CCIP;
    }

    public int getIncorrectly_Classified_Instances() {
        return Incorrectly_Classified_Instances;
    }

    public int getCorrectly_Classified_Instances() {
        return Correctly_Classified_Instances;
    }

    public double getICIP() {
        return ICIP;
    }

    public double getF_Measure() {
        return F_Measure;
    }

    public double getROC_Area() {
        return ROC_Area;
    }

    public String getReduction() {
        return reduction;
    }

    public int getReductionCount() {
        return reductionCount;
    }

    public String getDataSetName() {
        return dataSetName;
    }

    public boolean isReduced() {
        return isReduced;
    }

    public int getNumAttribute() {
        return numAttribute;
    }

    public static String header() {
        return "DataSet Name,delta,alpha,beta,Correctly Classified Instances,Correctly Classified Instances%," +
                "Incorrectly Classified Instances,Incorrectly Classified Instances%,F-Measure,AUC," +
                "(Reduction) Attribute Count,Reduction";
    }

    public String toString() {
        if (isReduced) {
            return dataSetName + "," + delta + "," + alpha + "," + beta + ","
                    + Correctly_Classified_Instances + "," + CCIP + "%,"
                    + Incorrectly_Classified_Instances + "," + ICIP + "%,"
                    + F_Measure + "," + ROC_Area + "," + reductionCount + "," + reduction.replaceAll(",","");
        } else {
            return dataSetName + "," + nullSymbol + "," + nullSymbol + "," + nullSymbol + ","
                    + Correctly_Classified_Instances + "," + CCIP + "%,"
                    + Incorrectly_Classified_Instances + "," + ICIP + "%,"
                    + F_Measure + "," + ROC_Area + "," + numAttribute ;
        }
    }
}
