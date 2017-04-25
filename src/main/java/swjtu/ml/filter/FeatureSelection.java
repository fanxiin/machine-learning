package swjtu.ml.filter;

import weka.core.*;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

import java.util.ArrayList;

/**
 * Created by xin on 2017/4/11.
 */
public class FeatureSelection extends Filter implements SupervisedFilter,
    OptionHandler, WeightedInstancesHandler {

    private FSAlgorithm m_FSAlgorithm;

    /** holds the selected attributes */
    private int[] m_SelectedAttributes;

    /** True if a class attribute is set in the data */
    protected boolean m_hasClass;

    public FeatureSelection(FSAlgorithm fsAlgorithm){
        m_FSAlgorithm = fsAlgorithm;
    }

    /**
     * Input an instance for filtering. Ordinarily the instance is processed and
     * made available for output immediately. Some filters require all instances
     * be read before producing output.
     *
     * @param instance the input instance
     * @return true if the filtered instance may now be collected with output().
     * @throws IllegalStateException if no input format has been defined.
     * @throws Exception if the input instance was not of the correct format or if
     *           there was a problem with the filtering.
     */
    @Override
    public boolean input(Instance instance) throws Exception {

        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        if (m_NewBatch) {
            resetQueue();
            m_NewBatch = false;
        }

        /**过滤器已经训练完成，后来的数据直接转换加入输出队列*/
        if (isOutputFormatDefined()) {
            convertInstance(instance);
            return true;
        }

        bufferInput(instance);
        return false;
    }

    /**
     * Signify that this batch of input to the filter is finished. If the filter
     * requires all instances prior to filtering, output() may now be called to
     * retrieve the filtered instances.
     *
     * @return true if there are instances pending output.
     * @throws IllegalStateException if no input structure has been defined.
     * @throws Exception if there is a problem during the attribute selection.
     */
    @Override
    public boolean batchFinished() throws Exception {

        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        if (!isOutputFormatDefined()) {
            m_hasClass = (getInputFormat().classIndex() >= 0);

            m_SelectedAttributes = m_FSAlgorithm.SelectAttributes(getInputFormat());

            if (m_SelectedAttributes == null) {
                throw new Exception("No selected attributes\n");
            }

            setOutputFormat();

            // Convert pending input instances
            for (int i = 0; i < getInputFormat().numInstances(); i++) {
                convertInstance(getInputFormat().instance(i));
            }
            flushInput();
        }

        m_NewBatch = true;
        return (numPendingOutput() != 0);
    }

    /**
     * Set the output format. Takes the currently defined attribute set
     * m_InputFormat and calls setOutputFormat(Instances) appropriately.
     *
     * @throws Exception if something goes wrong
     */
    protected void setOutputFormat() throws Exception {
        Instances informat = getInputFormat();

        if (m_SelectedAttributes == null) {
            setOutputFormat(null);
            return;
        }

        ArrayList<Attribute> attributes =
                new ArrayList<Attribute>(m_SelectedAttributes.length);

        int i;

        for (i = 0; i < m_SelectedAttributes.length; i++) {
            attributes.add((Attribute) informat.attribute(m_SelectedAttributes[i])
                    .copy());
        }

        Instances outputFormat =
                new Instances(getInputFormat().relationName(), attributes, 0);

        // if (!(m_ASEvaluator instanceof UnsupervisedSubsetEvaluator)
        // && !(m_ASEvaluator instanceof UnsupervisedAttributeEvaluator)) {
        if (m_hasClass) {
            outputFormat.setClassIndex(m_SelectedAttributes.length - 1);
        }

        setOutputFormat(outputFormat);
    }

    /**
     * Convert a single instance over. Selected attributes only are transfered.
     * The converted instance is added to the end of the output queue.
     *
     * @param instance the instance to convert
     * @throws Exception if something goes wrong
     */
    protected void convertInstance(Instance instance) throws Exception {
        double[] newVals = new double[getOutputFormat().numAttributes()];

        for (int i = 0; i < m_SelectedAttributes.length; i++) {
            int current = m_SelectedAttributes[i];
            newVals[i] = instance.value(current);
        }

        if (instance instanceof SparseInstance) {
            push(new SparseInstance(instance.weight(), newVals));
        } else {
            push(new DenseInstance(instance.weight(), newVals));
        }
    }

    /**
     * 以字符串的方式返回训练结果
     * @param data
     * @return
     * @throws Exception
     */
    public String selectFeature(Instances data) throws Exception{
        for (int i = 0; i < data.numInstances(); i++) {
            this.input(data.instance(i));
        }
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        if (!isOutputFormatDefined()) {
            m_hasClass = (getInputFormat().classIndex() >= 0);
            m_SelectedAttributes = m_FSAlgorithm.SelectAttributes(getInputFormat());
            if (m_SelectedAttributes == null) {
                throw new Exception("No selected attributes\n");
            }
            setOutputFormat();
            flushInput();
        }
        m_NewBatch = true;
        return m_FSAlgorithm.getSelectedAttributesString();
    }

}

