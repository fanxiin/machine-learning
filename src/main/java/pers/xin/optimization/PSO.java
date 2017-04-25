package pers.xin.optimization;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by xin on 2017/4/18.
 */
public class PSO {

    /** 优化参数个数 */
    private int numParam;

    /** 种群大小（粒子个数） */
    private int swarmSize;

    private int maxIteration;

    /** 惯性权重 */
    private double w;

    /** 学习因子 */
    private double c1=2,c2=2;

    /** 迭代停止阈值（最佳参数更新后与上一次迭代之间的欧式距离） */
    private double threshold;

    /** 粒子数组 */
    private ArrayList<Particle> m_particles;

    /** 所有粒子的最优点 */
    private double[] gBestPosition;

    /** 群体最佳适应度值 */
    private double gBestFitness;

    private Optimizable m_object;

    private double intervals[][];

    /**
     * 设置优化对象，并记录参数取值区间
     * @param m_object
     */
    public void setObject(Optimizable m_object) {
        this.numParam = m_object.getInterval().length;
        this.m_object = m_object;
        this.intervals = m_object.getInterval();
    }

    public PSO(int swarmSize, int maxIteration, double threshold, double w, double c1, double c2){
        this.swarmSize=swarmSize;
        this.maxIteration=maxIteration;
        this.threshold = threshold;
        this.w=w;
        this.c1=c1;
        this.c2=c2;
    }

    /**
     * 粒子类
     */
    class Particle{
        /** 粒子当前所在位置 */
        protected double[] position;

        /** 粒子当前速度 */
        protected double[] velocity;

        /** 粒子自己找到的最佳位置 */
        protected double[] pBestPosition;

        /** 粒子自身的历史最佳适应度 */
        protected double pBestFitness;

        private Random rand = new Random();

        public Particle(){
            position = new double[numParam];
            velocity = new double[numParam];
            pBestPosition = new double[numParam];

            for(int i=0; i<numParam; i++){
                position[i] = intervalRand(i);
                velocity[i] = velocityRand(i);
                pBestPosition[i] = position[i];
            }
            pBestFitness = m_object.fitness(position);
        }

        /**
         * 生成区间内的随机点
         * @param paramIndex
         * @return
         */
        private double intervalRand(int paramIndex){
            double[] interval = intervals[paramIndex];
            return interval[0]+rand.nextDouble()*(interval[1]-interval[0]);
        }

        /**
         * 生成初始速度
         * @param paramIndex
         * @return
         */
        private double velocityRand(int paramIndex){
            double[] interval = intervals[paramIndex];
            double v_upper = interval[1]-interval[0];
            return -v_upper+rand.nextDouble()*2*v_upper;
        }

        /**
         * 更新粒子速度
         */
        protected void updateVelocity(){
            for(int i=0; i<numParam; i++){
                velocity[i] = w*velocity[i]+c1* rand.nextDouble()*(pBestPosition[i]-position[i])
                        +c2*rand.nextDouble()*(gBestPosition[i] - position[i]);
            }
        }

        /**
         * 更新粒子位置
         */
        protected double updatePosition(){
            updateVelocity();
            for (int i = 0; i < numParam; i++) {
                position[i] = position[i] + velocity[i];
            }
//            limitPosition();
//            if(!inRange()){
//                for(int i=0; i<numParam; i++){
//                    position[i] = intervalRand(i);
//                    velocity[i] = velocityRand(i);
//                }
//            }
            return updateFitness();
        }

        protected double updateFitness(){
            double fitness = m_object.fitness(position);
            if(fitness< pBestFitness){
                pBestFitness = fitness;
                pBestPosition = position.clone();
            }
            return pBestFitness;
        }

        protected boolean inRange(){
            for (int i = 0; i < intervals.length; i++) {
                if(position[i]>intervals[i][1]||position[i]<intervals[i][0])
                    return false;
            }
            return true;
        }

        protected void limitPosition(){
            for (int i = 0; i < intervals.length; i++) {
                if(position[i]>intervals[i][1]){
                    position[i] = intervals[i][1];
                    velocity[i] = 0;
                }
                else if(position[i]<intervals[i][0]){
                    position[i] = intervals[i][0];
                    velocity[i] = 0;
                }
            }
        }
    }

    /**
     * 初始化粒子,及粒子群的群体最优点
     */
    private void initParticles(){
        m_particles = new ArrayList<Particle>();
        gBestFitness = Double.MAX_VALUE;
        for(int i=0;i<swarmSize;i++){
            Particle p = new Particle();
            m_particles.add(p);
            if(p.pBestFitness<gBestFitness){
                gBestFitness = p.pBestFitness;
                gBestPosition = p.pBestPosition.clone();
            }
        }
    }

    private double[] computeMidPoint(double[] first, double[] secend){
        double[] result = new double[first.length];
        for (int i = 0; i < first.length; i++) {
            result[i] = (first[i]+secend[i])/2.0;
        }
        return result;
    }

    private double distance(double[] first, double[] second){
        double distance = 0.0;
        for (int i = 0; i < first.length; i++) {
            distance += Math.pow(first[i]-second[i],2);
        }
        return Math.sqrt(distance);
    }

    /**
     * 搜索最优参数
     * @return
     */
    public double[] search(){
        initParticles();

        for(int i=0; i<maxIteration; i++){
            System.out.print(".");
            /** 此轮迭代所有粒子的最优点 */
            double[] m_gBestPosition = gBestPosition.clone();
            /** 此轮迭代群体最佳适应度值 */
            double m_gBestFitness = Double.MAX_VALUE;
            for(Particle p : m_particles){
                double pBestFitness = p.updatePosition();
                if(pBestFitness<m_gBestFitness){
                    m_gBestFitness = pBestFitness;
                    m_gBestPosition = p.pBestPosition.clone();
                }
            }
//            if(m_gBestFitness==gBestFitness){
//                double distance = distance(m_gBestPosition,gBestPosition);
//                if(distance<=threshold){
//                    return gBestPosition;
//                }
//                else {
//                    gBestPosition = computeMidPoint(m_gBestPosition,gBestPosition);
//                }
//            }
            if(m_gBestFitness<=gBestFitness){
                gBestPosition = m_gBestPosition;
                gBestFitness = m_gBestFitness;
            }
        }
        return gBestPosition;
    }

    public double bestFitness(){
        return gBestFitness;
    }

}


