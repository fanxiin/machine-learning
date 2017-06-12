package pers.xin.optimization;

import org.apache.log4j.*;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Random;

/**
 * Created by xin on 2017/4/18.
 */
public class PSO {

    private Logger logger;
    private Logger m_logger;

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

    private int[] precision;

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
    private Fitness gBestFitness;

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
        this.precision= m_object.getPrecision();
        vMax = new double[intervals.length][2];
        for (int i = 0; i < vMax.length; i++) {
            vMax[i][1]= (intervals[i][1]-intervals[i][0])*velocityRatio;
            vMax[i][0]= -vMax[i][1];
        }
    }

    public void setLogger(Logger logger, Logger m_logger) throws IOException {
        this.logger = logger;
        this.m_logger = m_logger;
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
    public PSO(int swarmSize, int maxIteration, double velocityRatio, double threshold, double w,
               double c1, double
            c2) {
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
        protected Fitness pBestFitness;

        private Random rand = new Random();

        public Particle() {
            position = new double[numParam];
            velocity = new double[numParam];
            pBestPosition = new double[numParam];

            for (int i = 0; i < numParam; i++) {
                position[i] = intervalRand(i);
                BigDecimal bg = new BigDecimal(position[i]);
                if(precision!=null){
                    position[i] = bg.setScale(precision[i],BigDecimal.ROUND_HALF_UP).doubleValue();
                }
                velocity[i] = velocityRand(i);
                pBestPosition[i] = position[i];
            }
//            pBestFitness = Double.MAX_VALUE;
            pBestFitness = m_object.computeFitness(position);
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
            updateVelocity();
            for (int i = 0; i < numParam; i++) {
                position[i] = position[i] + velocity[i];
                BigDecimal bg = new BigDecimal(position[i]);
                if(precision!=null){
                    position[i] = bg.setScale(precision[i],BigDecimal.ROUND_HALF_UP).doubleValue();
                }
            }
            limitPosition();
        }

        protected Fitness getFitness() {
            Fitness fitness = m_object.computeFitness(position);
            if (fitness.isBetterThan(pBestFitness)) {
                pBestFitness = fitness;
                pBestPosition = position.clone();
            }
            StringBuilder sb = new StringBuilder();
            for (double v : position) {
                sb.append(v+" ");
            }
            sb.append(fitness.fitness());
            logger.info(sb.toString());
            m_logger.info(sb.toString());
            return pBestFitness;
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
        gBestFitness = null;
        for (int i = 0; i < swarmSize; i++) {
            Particle p = new Particle();
            m_particles.add(p);
            if (p.pBestFitness.isBetterThan(gBestFitness)) {
                gBestFitness = p.pBestFitness;
                gBestPosition = p.pBestPosition.clone();
            }
        }
    }

    /**
     * 搜索最优参数
     *
     * @return
     */
    public double[] search() {
        initParticles();

        for (int i = 0; i < maxIteration; i++) {
            logger.info("iterate "+i);
            System.out.print(".");
            /** 此轮迭代所有粒子的最优点 */
            double[] m_gBestPosition = gBestPosition.clone();
            /** 此轮迭代群体最佳适应度值 */
            Fitness m_gBestFitness = null;
            for (Particle p : m_particles) {
                Fitness pBestFitness = p.getFitness();
                p.updatePosition();
                if (pBestFitness.isBetterThan(m_gBestFitness)) {
                    m_gBestFitness = pBestFitness;
                    m_gBestPosition = p.pBestPosition.clone();
                }
            }
            if (m_gBestFitness.isBetterThan(gBestFitness)) {
                gBestPosition = m_gBestPosition;
                gBestFitness = m_gBestFitness;
            }
        }
        return gBestPosition;
    }

    public void reset(){
        initParticles();
    }

    public Fitness bestFitness() {
        return gBestFitness;
    }

}


