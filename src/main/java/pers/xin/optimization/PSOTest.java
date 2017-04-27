package pers.xin.optimization;

import weka.core.pmml.Array;

import java.util.ArrayList;
import java.util.HashSet;

/**
 * Created by xin on 2017/4/19.
 */
public class PSOTest implements Optimizable{
    public double fitness(double[] params) {
        return 3*Math.pow(params[0]-1,2)+2*Math.pow(params[1]+3,2)+3*Math.pow(params[2]+10,2);
    }

    public double[][] getInterval() {
        double[][] m_interval = {{0,20},{-4,20},{-10,90}};
        return m_interval;
    }

    public int paramCount() {
        return 3;
    }

    public static void main(String[] args) {
        PSO pso = new PSO(10,200,0.1,0.0001,0.5,2,2);
        pso.setObject(new PSOTest());
        double[] best = pso.search();
        System.out.println(best[0]+" ,"+best[1]+", "+best[2]);
        System.out.println(pso.bestFitness());
        HashSet<Integer> hs = new HashSet<Integer>();
        hs.add(1);
        hs.add(new Integer(1));
        hs.add(2-1);
        hs.add(2);
    }
}
