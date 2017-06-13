package pers.xin.optimization;

import java.util.HashSet;

/**
 * Created by xin on 2017/4/19.
 */
public class PSOTest extends Optimizable{
    public Fitness computeFitness(double[] params) {
        return new BaseFitness(3*Math.pow(params[0]-1,2)+2*Math.pow(params[1]+3,2)+3*Math.pow(params[2]+10,2));
    }

    public int[] getPrecision() {
        int[] p = {3,3,3};
        return p;
    }

    public double[][] getInterval() {
        double[][] m_interval = {{0,20},{-4,20},{-10,90}};
        return m_interval;
    }

    public int paramCount() {
        return 3;
    }

    public static void main(String[] args) {
        BPSO pso = new BPSO(20,40,0.8,0.0001,0.5,2,2);
        pso.setObject(new PSOTest());
        double[] best = pso.search();
        System.out.println(best[0]+" ,"+best[1]+", "+best[2]);
        System.out.println(((BaseFitness)pso.bestFitness()).m_fitness);
        HashSet<Integer> hs = new HashSet<Integer>();
        hs.add(1);
        hs.add(new Integer(1));
        hs.add(2-1);
        hs.add(2);
    }
}
