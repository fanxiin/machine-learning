package pers.xin.optimization;

import org.junit.Assert;

import static org.junit.Assert.*;

/**
 * Created by xin on 2017/4/27.
 */
public class AUCFSFitnessTest {
    @org.junit.Test
    public void isBetterThan() throws Exception {
        Fitness a = new AUCFSFitness(0.95,8);
        Fitness b = new AUCFSFitness(0.95,9);
        Fitness c = new AUCFSFitness(0.8,2);
        Assert.assertTrue(a.isBetterThan(b));
        Assert.assertTrue(a.isBetterThan(c));
        Assert.assertTrue(b.isBetterThan(c));
        Assert.assertFalse(b.isBetterThan(a));
        Assert.assertFalse(c.isBetterThan(a));
        Assert.assertFalse(c.isBetterThan(b));
    }

}