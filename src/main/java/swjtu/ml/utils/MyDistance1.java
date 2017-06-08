package swjtu.ml.utils;

import weka.core.*;
import weka.core.neighboursearch.PerformanceStats;

/**
 * Created by xin on 2017/4/13.
 * 全部为离散型变量时，距离为0表示变量取值相等
 */
public class MyDistance1 extends NormalizableDistance {

    private int numNominal;

    private int numNumeric;

    public MyDistance1(){super();}

    public MyDistance1(Instances data){
        super(data);
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            if (data.attribute(i).type() == Attribute.NUMERIC) {
                numNumeric++;
            }
            if (data.attribute(i).type() == Attribute.NOMINAL) {
                numNominal++;
            }
        }
    }

    public String globalInfo() {
        return null;
    }

    protected double[] updateDistance(double[] currDist, double[] diff) {
        double[]	result;

        result  = currDist;
        result[0] += diff[0] * diff[0];

        result[1] += diff[1];

        return result;
    }

    protected double updateDistance(double currDist, double diff) {
        double	result;

        result  = currDist;
        result += diff * diff;

        return result;
    }

    protected double[] m_difference(int index, double val1, double val2){
        switch (m_Data.attribute(index).type()) {
            case Attribute.NOMINAL:
                if (Utils.isMissingValue(val1) || Utils.isMissingValue(val2)
                        || ((int) val1 != (int) val2)) {
                    return new double[]{0,1};
                } else {
                    return new double[]{0,0};
                }

            case Attribute.NUMERIC:
                if (Utils.isMissingValue(val1) || Utils.isMissingValue(val2)) {
                    if (Utils.isMissingValue(val1) && Utils.isMissingValue(val2)) {
                        if (!m_DontNormalize) {
                            return new double[]{1,0};
                        } else {
                            return new double[]{(m_Ranges[index][R_MAX] - m_Ranges[index][R_MIN]),0};
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
                                return new double[]{m_Ranges[index][R_MAX] - diff,0};
                            } else {
                                return new double[]{diff - m_Ranges[index][R_MIN],0};
                            }
                        }
                        return new double[]{diff,0};
                    }
                } else {
                    double diff = (!m_DontNormalize) ? (norm(val1, index) - norm(val2, index))
                            : (val1 - val2);
                    return new double[]{diff,0};
                }

            default:
                return new double[]{0,0};
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
    public double[] multipleDistance(Instance first, Instance second, double cutOffValue,
                           PerformanceStats stats) {
        double distance[] = {0,0};
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

            double diff[];

            if (firstI == secondI) {
                diff = m_difference(firstI, first.valueSparse(p1), second.valueSparse(p2));
                p1++;
                p2++;
            } else if (firstI > secondI) {
                diff = m_difference(secondI, 0, second.valueSparse(p2));
                p2++;
            } else {
                diff = m_difference(firstI, first.valueSparse(p1), 0);
                p1++;
            }
            if (stats != null) {
                stats.incrCoordCount();
            }

            distance = updateDistance(distance, diff);
            if (distance[0] > cutOffValue) {
                return new double[]{Double.POSITIVE_INFINITY,0};
            }
        }

        return distance;
    }

    @Override
    public double distance(Instance first, Instance second) {
        return Math.sqrt(super.distance(first, second));
    }

    public double[] multipleDistance(Instance first, Instance second){

        double[] distance = multipleDistance(first, second, Double.POSITIVE_INFINITY, null);
        if(numNumeric!=0)distance[0]=distance[0]/numNumeric;
        if(numNominal!=0)distance[1]=distance[1]/numNominal;

        return distance;
    }

    public String getRevision() {
        return null;
    }
}
