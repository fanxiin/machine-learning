package pers.xin.Experiment;

import weka.classifiers.Evaluation;

/**
 * Created by xin on 2017/4/25.
 */
public class FormatSummary {
    private double[] params;
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
    private String nullSymbol = "";
    private int paramCount=0;

    /**
     *
     * @param nullSymbol
     * @param paramCount
     */
    public FormatSummary(String nullSymbol, int paramCount){
        this.nullSymbol = nullSymbol;
        this.paramCount=paramCount;
    }

    public void setSummary(Evaluation eval, int positiveIndex) {
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

    public void setSummary(double[] params, String reduction, Evaluation eval, int positiveIndex) {
        setSummary(eval, positiveIndex);
        this.params = params;
        this.reduction = reduction;
        this.isReduced = true;
        this.reductionCount = reduction.split(",").length;
    }

    public double getROC_Area() {
        return ROC_Area;
    }

    public String getReduction() {
        return reduction;
    }

    public String header() {
        StringBuilder sb = new StringBuilder();
        sb.append("DataSet Name,");
        for (int i = 0; i < paramCount; i++) {
            sb.append("parma"+(i+1)+",");
        }
        sb.append("Correctly Classified Instances,Correctly Classified Instances%," +
                "Incorrectly Classified Instances,Incorrectly Classified Instances%,F-Measure,AUC," +
                        "(Reduction) Attribute Count,Reduction");
        return sb.toString();
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (isReduced) {
            sb.append(dataSetName + ",");
            for (int i = 0; i < paramCount; i++) {
                sb.append(params[i]+",");
            }
            sb.append(Correctly_Classified_Instances + "," + CCIP + "%,"
                    + Incorrectly_Classified_Instances + "," + ICIP + "%,"
                    + F_Measure + "," + ROC_Area + "," + reductionCount + "," + reduction.replaceAll(",",""));
            return sb.toString();
        } else {
            sb.append(dataSetName + ",");
            for (int i = 0; i < paramCount; i++) {
                sb.append(nullSymbol+",");
            }
            sb.append(Correctly_Classified_Instances + "," + CCIP + "%,"
                    + Incorrectly_Classified_Instances + "," + ICIP + "%,"
                    + F_Measure + "," + ROC_Area + "," + numAttribute);
            return sb.toString();
        }
    }

    public String format(Evaluation eval, int positiveIndex) {
        setSummary(eval,positiveIndex);
        return toString();
    }

    public String format(double[] params, String reduction, Evaluation eval, int positiveIndex) {
        setSummary(params,reduction,eval,positiveIndex);
        return toString();
    }
}
