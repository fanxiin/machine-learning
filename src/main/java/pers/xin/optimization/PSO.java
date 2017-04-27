package pers.xin.optimization;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;

/**
 * Created by xin on 2017/4/18.
 */
public class PSO {

    /**
     * 优化参数个数
     */
    private int numParam;

    /**
     * 种群大小（粒子个数）
     */
    private int swarmSize;

    /** 最大迭代次数 */
    private int maxIteration;

    /**
     * 惯性权重
     */
    private double w;

    /**
     * 学习因子
     */
    private double c1 = 2, c2 = 2;

    /**
     * 迭代停止阈值
     */
    private double threshold;

    /**
     * 粒子数组
     */
    private ArrayList<Particle> m_particles;

    /**
     * 所有粒子的最优点
     */
    private double[] gBestPosition;

    /**
     * 群体最佳适应度值
     */
    private double gBestFitness;

    /** 优化对象，给出适应度函数 */
    private Optimizable m_object;

    /** 搜索区间 */
    private double intervals[][];

    private double velocityRatio;

    /** 粒子最大速度 */
    private double vMax[][];

    /**
     * 设置优化对象，并记录参数取值区间
     *
     * @param m_object
     */
    public void setObject(Optimizable m_object) {
        this.numParam = m_object.getInterval().length;
        this.m_object = m_object;
        this.intervals = m_object.getInterval();
        vMax = new double[intervals.length][2];
        for (int i = 0; i < vMax.length; i++) {
            vMax[i][1]= (intervals[i][1]-intervals[i][0])*velocityRatio;
            vMax[i][0]= -vMax[i][1];
        }
    }

    /**
     * 初始化一个粒子群
     * @param swarmSize 种群大小
     * @param maxIteration 最大迭代次数
     * @param velocityRatio 最大速度与搜索区间的比值
     * @param threshold 停止阈值
     * @param w 惯性权重
     * @param c1 自我认知系数
     * @param c2 社会认知系数
     */
    public PSO(int swarmSize, int maxIteration,double velocityRatio, double threshold, double w, double c1, double c2) {
        this.swarmSize = swarmSize;
        this.maxIteration = maxIteration;
        this.velocityRatio = velocityRatio;
        this.threshold = threshold;
        this.w = w;
        this.c1 = c1;
        this.c2 = c2;
    }

    /**
     * 粒子类
     */
    class Particle {
        /**
         * 粒子当前所在位置
         */
        protected double[] position;

        /**
         * 粒子当前速度
         */
        protected double[] velocity;

        /**
         * 粒子自己找到的最佳位置
         */
        protected double[] pBestPosition;

        /**
         * 粒子自身的历史最佳适应度
         */
        protected double pBestFitness;

        private Random rand = new Random();

        public Particle() {
            position = new double[numParam];
            velocity = new double[numParam];
            pBestPosition = new double[numParam];

            for (int i = 0; i < numParam; i++) {
                position[i] = intervalRand(i);
                velocity[i] = velocityRand(i);
                pBestPosition[i] = position[i];
            }
            pBestFitness = m_object.fitness(position);
        }

        /**
         * 生成区间内的随机点
         *
         * @param paramIndex
         * @return
         */
        private double intervalRand(int paramIndex) {
            double[] interval = intervals[paramIndex];
            return interval[0] + rand.nextDouble() * (interval[1] - interval[0]);
        }

        /**
         * 生成初始速度
         *
         * @param paramIndex
         * @return
         */
        private double velocityRand(int paramIndex) {
            return vMax[paramIndex][0] + rand.nextDouble() *(vMax[paramIndex][1]-vMax[paramIndex][0]);
        }

        /**
         * 更新粒子速度
         */
        private void updateVelocity() {
            for (int i = 0; i < numParam; i++) {
                velocity[i] = w * velocity[i] + c1 * rand.nextDouble() * (pBestPosition[i] - position[i])
                        + c2 * rand.nextDouble() * (gBestPosition[i] - position[i]);
            }
            limitVelocity();
        }

        /**
         * 更新粒子位置
         */
        private void updatePosition() {
            for (int i = 0; i < numParam; i++) {
                position[i] = position[i] + velocity[i];
            }
            limitPosition();
//            if(!inRange()){
//                for(int i=0; i<numParam; i++){
//                    position[i] = intervalRand(i);
//                    velocity[i] = velocityRand(i);
//                }
//            }
        }

        protected double updateFitness() {
            double fitness = m_object.fitness(position);
            if (fitness < pBestFitness) {
                pBestFitness = fitness;
                pBestPosition = position.clone();
            }
            System.out.println("("+position[0]+","+position[1]+","+position[2]+")");
            updateVelocity();
            updatePosition();
            return pBestFitness;
        }

        protected boolean inRange() {
            for (int i = 0; i < intervals.length; i++) {
                if (position[i] > intervals[i][1] || position[i] < intervals[i][0])
                    return false;
            }
            return true;
        }

        protected void limitPosition() {
            for (int i = 0; i < intervals.length; i++) {
                if (position[i] > intervals[i][1]) position[i] = intervals[i][1];
                else if (position[i] < intervals[i][0]) position[i] = intervals[i][0];
            }
        }

        protected void limitVelocity() {
            for (int i = 0; i < vMax.length; i++) {
                if (velocity[i] > vMax[i][1]) {
                    velocity[i] = vMax[i][1];
                } else if (velocity[i] < vMax[i][0]) {
                    velocity[i] = vMax[i][0];
                }
            }
        }
    }

    /**
     * 初始化粒子,及粒子群的群体最优点
     */
    private void initParticles() {
        m_particles = new ArrayList<Particle>();
        gBestFitness = Double.MAX_VALUE;
        for (int i = 0; i < swarmSize; i++) {
            Particle p = new Particle();
            m_particles.add(p);
            if (p.pBestFitness < gBestFitness) {
                gBestFitness = p.pBestFitness;
                gBestPosition = p.pBestPosition.clone();
            }
        }
    }

    private double[] computeMidPoint(double[] first, double[] secend) {
        double[] result = new double[first.length];
        for (int i = 0; i < first.length; i++) {
            result[i] = (first[i] + secend[i]) / 2.0;
        }
        return result;
    }

    private double distance(double[] first, double[] second) {
        double distance = 0.0;
        for (int i = 0; i < first.length; i++) {
            distance += Math.pow(first[i] - second[i], 2);
        }
        return Math.sqrt(distance);
    }

    /**
     * 搜索最优参数
     *
     * @return
     */
    public double[] search() {
        initParticles();

        for (int i = 0; i < maxIteration; i++) {
//            System.out.print(".");
            /** 此轮迭代所有粒子的最优点 */
            double[] m_gBestPosition = gBestPosition.clone();
            /** 此轮迭代群体最佳适应度值 */
            double m_gBestFitness = Double.MAX_VALUE;
            for (Particle p : m_particles) {
                double pBestFitness = p.updateFitness();
                if (pBestFitness < m_gBestFitness) {
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
            if (m_gBestFitness <= gBestFitness) {
                gBestPosition = m_gBestPosition;
                gBestFitness = m_gBestFitness;
            }
        }
        return gBestPosition;
    }

    public double bestFitness() {
        return gBestFitness;
    }

}


