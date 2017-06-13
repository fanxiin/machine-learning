package pers.xin.optimization;

import org.apache.log4j.Logger;

import java.io.IOException;

/**
 * Created by xin on 2017/6/13.
 */
public interface PSO {
    void setObject(Optimizable m_object);
    void setLogger(Logger logger, Logger m_logger) throws IOException;
    double[] search();
}
