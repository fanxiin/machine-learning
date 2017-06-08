package swjtu.ml.utils;

import weka.core.*;
import weka.core.neighboursearch.PerformanceStats;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by xin on 2017/4/13.
 * 全部为离散型变量时，距离为0表示变量取值相等
 */
public class HVDM1 extends NormalizableDistance {

    private HashMap<Integer,ArrayList<ArrayList<Double>>> conditionProbabilities;

    public HVDM1(){super();}

    public HVDM1(Instances data){
        super(data);
        int numClass = data.numClasses();
        conditionProbabilities = new HashMap<Integer, ArrayList<ArrayList<Double>>>();
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            if (data.attribute(i).type() == Attribute.NOMINAL) {
                int[][] counts = new int[data.attribute(i).numValues()][numClass];
                int[] valueCounts = new int[data.attribute(i).numValues()];
                int numAttrValues = data.attribute(i).numValues();
                for (Instance datum : data) {
                    counts[(int)datum.value(i)][(int)datum.classValue()]++;
                    valueCounts[(int)datum.value(i)]++;
                }
                ArrayList<ArrayList<Double>> probability = new ArrayList<ArrayList<Double>>();

                for (int j = 0; j < numAttrValues; j++) {
                    ArrayList<Double> tmp = new ArrayList<Double>();
                    for (int k = 0; k < numClass; k++) {
                        tmp.add(k,counts[j][k]*1.0/valueCounts[j]);
                    }
                    probability.add(j,tmp);
                }
                conditionProbabilities.put(i,probability);
            }
        }
    }

    public String globalInfo() {
        return null;
    }

    protected double updateDistance(double currDist, double diff) {
        double	result;

        result  = currDist;
        result += diff * diff;

        return result;
    }

    @Override
    protected double difference(int index, double val1, double val2){
        switch (m_Data.attribute(index).type()) {
            case Attribute.NOMINAL:

                if (Utils.isMissingValue(val1) || Utils.isMissingValue(val2)) {
                    return 1;
                } else {
                    if((int) val1 == (int) val2) return 0;
                    else {
                        int numClass = m_Data.numClasses();
                        ArrayList<ArrayList<Double>> probabilities = conditionProbabilities.get(index);
                        double diff=0.0;
                        for (int i = 0; i < numClass; i++) {
                            diff += Math.pow(probabilities.get((int)val1).get(i) - probabilities.get((int)val2).get
                                    (i),2);
                        }
                        return Math.sqrt(diff);
                    }
                }

            case Attribute.NUMERIC:
                if (Utils.isMissingValue(val1) || Utils.isMissingValue(val2)) {
                    if (Utils.isMissingValue(val1) && Utils.isMissingValue(val2)) {
                        if (!m_DontNormalize) {
                            return 1;
                        } else {
                            return (m_Ranges[index][R_MAX] - m_Ranges[index][R_MIN]);
                        }
                    } else {
                        double diff;
                        if (Utils.isMissingValue(val2)) {
                            diff = (!m_DontNormalize) ? norm(val1, index) : val1;
                        } else {
                            diff = (!m_DontNormalize) ? norm(val2, index) : val2;
                        }
                        if (!m_DontNormalize && diff < 0.5) {
                            diff = 1.0 - diff;
                        } else if (m_DontNormalize) {
                            if ((m_Ranges[index][R_MAX] - diff) > (diff - m_Ranges[index][R_MIN])) {
                                return m_Ranges[index][R_MAX] - diff;
                            } else {
                                return diff - m_Ranges[index][R_MIN];
                            }
                        }
                        return diff;
                    }
                } else {
                    return norm(val1, index) - norm(val2, index);
                }

            default:
                return 0;
        }
    }

    /**
     * 返回两个实例在单个属性上的距离
     * @param index
     * @param first
     * @param second
     * @return
     */
    public double distanceOnAttr(int index, Instance first, Instance second){
        return Math.abs(difference(index,first.value(index),second.value(index)));
    }

    /**
     * Calculates the distance between two instances. Offers speed up (if the
     * distance function class in use supports it) in nearest neighbour search by
     * taking into account the cutOff or maximum distance. Depending on the
     * distance function class, post processing of the distances by
     * postProcessDistances(double []) may be required if this function is used.
     *
     * @param first the first instance
     * @param second the second instance
     * @param cutOffValue If the distance being calculated becomes larger than
     *          cutOffValue then the rest of the calculation is discarded.
     * @param stats the performance stats object
     * @return the distance between the two given instances or
     *         Double.POSITIVE_INFINITY if the distance being calculated becomes
     *         larger than cutOffValue.
     */
    @Override
    public double distance(Instance first, Instance second, double cutOffValue,
                           PerformanceStats stats) {
        double distance = 0;
        int firstI, secondI;
        int firstNumValues = first.numValues();
        int secondNumValues = second.numValues();
        int numAttributes = m_Data.numAttributes();
        int classIndex = m_Data.classIndex();

        validate();

        for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {
            if (p1 >= firstNumValues) {
                firstI = numAttributes;
            } else {
                firstI = first.index(p1);
            }

            if (p2 >= secondNumValues) {
                secondI = numAttributes;
            } else {
                secondI = second.index(p2);
            }

            if (firstI == classIndex) {
                p1++;
                continue;
            }
            if ((firstI < numAttributes) && !m_ActiveIndices[firstI]) {
                p1++;
                continue;
            }

            if (secondI == classIndex) {
                p2++;
                continue;
            }
            if ((secondI < numAttributes) && !m_ActiveIndices[secondI]) {
                p2++;
                continue;
            }

            double diff;

            if (firstI == secondI) {
                diff = difference(firstI, first.valueSparse(p1), second.valueSparse(p2));
                p1++;
                p2++;
            } else if (firstI > secondI) {
                diff = difference(secondI, 0, second.valueSparse(p2));
                p2++;
            } else {
                diff = difference(firstI, first.valueSparse(p1), 0);
                p1++;
            }
            if (stats != null) {
                stats.incrCoordCount();
            }

            if(diff >= Double.POSITIVE_INFINITY){
                return Double.POSITIVE_INFINITY;
            }

            distance = updateDistance(distance, diff);
            if (distance > cutOffValue) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return distance/(m_Data.numAttributes()-1);
    }

    @Override
    public double distance(Instance first, Instance second) {
        return Math.sqrt(super.distance(first, second));
    }

    public String getRevision() {
        return null;
    }
}
