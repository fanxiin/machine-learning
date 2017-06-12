package pers.xin.Experiment;

import weka.classifiers.Evaluation;

/**
 * Created by xin on 2017/4/25.
 */
public class FormatSummary {

    public static String[] paramNames;

    public static void setParamNames(String[] paramNames) {
        FormatSummary.paramNames = paramNames;
    }

    public static String format(Evaluation eval, int positiveIndex) {
        String result = (int) eval.correct()+ ","+
                eval.pctCorrect()+ ","+
                (int) eval.numTruePositives(positiveIndex)+ ","+
                eval.truePositiveRate(positiveIndex)*100+ ","+
                (int) eval.numTrueNegatives(positiveIndex)+ ","+
                eval.trueNegativeRate(positiveIndex)*100+ ","+
                eval.fMeasure(positiveIndex)+","+
                eval.areaUnderROC(positiveIndex)+","+
                eval.getHeader().numAttributes();
        return result;
    }

    public static String formatHeader(){
        return "correctly classified instances,correctly classified instances %," +
                "correctly classified positives,true positive rate %," +
                "correctly classified negatives,true positive rate %," +
                "F-Measure,AUC," +
                "(Reduction) Attribute Count";
    }

    public static String header(){
        String s_params = "";
        if(paramNames!=null){
            s_params = paramNames[0];
            for (int i = 1; i < paramNames.length; i++) {
                s_params = s_params +","+ paramNames[i];
            }
        }
        return "data set,"+s_params+","+formatHeader();
    }

    public static String format(Summary summary){
        Evaluation eval = summary.getEval();
        if(summary.isReducted()){
            double[] params = summary.getParams();
            String s_params = ""+params[0];
            for (int i = 1; i < params.length; i++) {
                s_params = s_params + ","+params[i];
            }
            int[] reduction = summary.getReduction();
            String s_reduction = "" + (reduction[0]+1);
            for (int i = 1; i < reduction.length-1; i++) {
                s_reduction = s_reduction +" "+ (reduction[i]+1);
            }
            return eval.getHeader().relationName()+","+
                    s_params+","+
                    format(eval,summary.getPositiveIndex())+","+
                    s_reduction;
        }else {
            String s_params = "";
            if(paramNames!=null){
                for (int i = 1; i < paramNames.length; i++) {
                    s_params = s_params +",";
                }
            }
            return eval.getHeader().relationName()+","+
                    s_params+","+
                    format(eval,summary.getPositiveIndex());
        }
    }
}
