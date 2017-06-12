package pers.xin.Experiment;

import org.apache.log4j.*;
import pers.xin.mian.Main;
import pers.xin.optimization.PSO;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;

import java.io.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

/**
 * Created by xin on 2017/6/12.
 */
public class ExperimentDriver {
    private Logger logger;
    private Logger m_logger;
    private int maxIterate;
    private int swarmSize;
    private int psoTimes;
    private String m_FSALgorithm;
    private String[] classifiers;
    private double[][] interval;
    private int[] precision;
    private String dataFilePath;
    private String m_outputPath;

    public void setPSO(int swarmSize,int maxIterate, int psoTimes) {
        this.maxIterate = maxIterate;
        this.swarmSize = swarmSize;
        this.psoTimes = psoTimes;
    }

    public void setFSALgorithm(String m_FSALgorithm) {
        this.m_FSALgorithm = m_FSALgorithm;
    }

    public void setClassifiers(String[] classifiers) {
        this.classifiers = classifiers;
    }

    public void setInterval(double[][] interval) {
        this.interval = interval;
    }

    public void setPrecision(int[] precision) {
        this.precision = precision;
    }

    public void setDataFilePath(String dataFilePath) {
        this.dataFilePath = dataFilePath;
    }

    public void setOutputPath(String outputPath) {
        Date date = new Date();
        DateFormat format = new SimpleDateFormat("MMddHHmm");
        this.m_outputPath = outputPath+"/"+
                m_FSALgorithm.substring(m_FSALgorithm.lastIndexOf(".")+1) +"/"+
                format.format(date);
        File file = new File(m_outputPath);
        if(!file.exists()){
            file.getParentFile().mkdirs();
        }
    }

    public void initLogger() throws IOException {
        logger = Logger.getLogger("detail");
        m_logger=Logger.getLogger("matlab");
        Layout layout = new PatternLayout("%m%n");
        Appender appender = new FileAppender(layout,m_outputPath+"/detail.log");
        logger.addAppender(appender);
        Layout m_layout = new PatternLayout("%m%n");
        Appender m_appender = new FileAppender(m_layout,m_outputPath+"/matlab.log");
        m_logger.addAppender(m_appender);
    }

    public void run() throws Exception {
        initLogger();
        File folder = new File(dataFilePath);
        File[] files = folder.listFiles();

        StringBuilder ss = new StringBuilder();
        for (String classifier : classifiers) {
            ss.append(classifier);
            ss.append("\n");
        }
        ss.append("----------------\n");
        for (File file : files) {
            String name = file.getName();
            if(!name.startsWith(".")){
                ss.append(name);
                ss.append("\n");
            }
        }
        ss.append("swarmSize:"+swarmSize+"\n");
        ss.append("MaxIterate:"+maxIterate+"\n");
        ss.append("PSOTimes:"+psoTimes+"\n");
        ss.append("----------------\n");
        m_logger.info(ss.toString());
        logger.info(ss.toString());

        for (String classifierName : classifiers) {
            resultPrintln(classifierName);
            resultPrintln(FormatSummary.header());
            for (File file : files) {
                if(!file.getName().startsWith(".")){
                    System.out.println("-------- 处理数据集: "+file.getName() +" ---------");
                    logger.info("-------- "+file.getName() +"-"+classifierName+" ---------");
                    try{
                        Instances instances = new Instances(new FileReader(file));
                        instances.setClassIndex(instances.numAttributes()-1);
//                        Instances t = m.smote(instances);
                        Experiment experiment = new Experiment(classifierName,instances,5);
                        experiment.setFSAlgorithmName(m_FSALgorithm);
                        experiment.setInterval(interval);
                        experiment.setPrecision(precision);
                        Summary originalSummary = experiment.originalAnalyze();
                        resultPrintln(FormatSummary.format(originalSummary));
                        //logger.info("-------- origin AUC"+summary.getROC_Area()+" ---------");
                        PSO pso = new PSO(swarmSize,maxIterate,1,0.00001,0.5,2,2);
                        pso.setObject(experiment);
                        pso.setLogger(logger,m_logger);
                        for (int i = 0; i < psoTimes; i++) {
                            logger.info("-------- pso"+i+" ---------");
                            double[] params = pso.search();
                            Summary FSSummary = experiment.FSAnalyze(params);
                            resultPrintln(FormatSummary.format(FSSummary));
                        }
                        resultPrintln("");
                    }catch (Exception e){
                        e.printStackTrace();
                        continue;
                    }
                }
            }
        }
    }

    public Instances smote(Instances instances) throws Exception {
        int count[] = new int[2];
        for (Instance instance : instances) {
            count[(int)instance.classValue()]++;
        }
        int percentage = count[0]>count[1] ? (count[0]-count[1])*100/count[1] : (count[1]-count[0])*100/count[0];
        SMOTE s = new SMOTE();
        int seed = (int) (Math.random() * 10);
        String[] options = {"-S", String.valueOf(seed), "-P", ""+percentage, "-K", "5"};
        s.setOptions(options);
        s.setInputFormat(instances);
        return Filter.useFilter(instances,s);
    }

    public void resultPrintln(String data) throws Exception{
        File file = new File(m_outputPath+"/"+
                m_FSALgorithm.substring(m_FSALgorithm.lastIndexOf(".")+1)+".csv");
        if(!file.getParentFile().exists()){
            file.getParentFile().mkdirs();
        }
        if(!file.exists()){
            file.createNewFile();
        }
        FileWriter fw = new FileWriter(file,true);
        PrintWriter pw = new PrintWriter(fw);
        pw.println(data);
        pw.close();
    }

}
