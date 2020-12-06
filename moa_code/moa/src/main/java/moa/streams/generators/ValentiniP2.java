package moa.streams.generators;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.*;
import moa.capabilities.CapabilitiesHandler;
import moa.core.Example;
import moa.core.FastVector;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

import java.util.Random;

public class ValentiniP2 extends AbstractOptionHandler implements
        InstanceStream, CapabilitiesHandler {

    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1);

    public FlagOption balanceClassesOption = new FlagOption("balanceClasses",
            'b', "Balance the number of instances of each class.");

    @Override
    public String getPurposeString() {
        return "Generates a dataset following Valentini's non-linear P2 classification problem.";
    }

    // The stream header
    protected InstancesHeader streamHeader;

    // the pseudo-random number generator
    protected Random instanceRandom;

    // flag for determining whether the next instance should be positive
    protected boolean nextInstanceShouldBeOne;

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        // generates the header
        FastVector attributes = new FastVector();
        attributes.addElement(new Attribute("x"));
        attributes.addElement(new Attribute("y"));
        FastVector classLabels = new FastVector();
        classLabels.addElement("groupA");
        classLabels.addElement("groupB");
        attributes.addElement(new Attribute("class", classLabels));
        this.streamHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
        restart();
    }

    @Override
    public Example<Instance> nextInstance() {

        double x = 0.0;
        double y = 0.0;
        double label = 0.0;

        boolean desiredClassFound = false;
        while(!desiredClassFound) {
            x = instanceRandom.nextDouble();
            y = instanceRandom.nextDouble();
            if (y > (-0.1 * Math.pow((x * 10), 2) +
                    0.6 * Math.sin(4 * x * 10) + 8.0) / 10.0
                    &&
                    y > (Math.pow(x * 10 - 2, 2) + 1) / 10.0
                    ||
                    y < (2 * Math.sin(x * 10) + 5) / 10.0
                            &&
                            y > (Math.pow(x * 10 - 2, 2) + 1) / 10.0
                    ||
                    y < (-0.1 * Math.pow(x * 10, 2) + 0.6 * Math.sin(4 * x * 10) + 8.0) / 10
                            && y < (Math.pow(x * 10 - 2, 2) + 1) / 10.0
                            &&
                            y > (2 * Math.sin(x * 10) + 5) / 10.0
                    ||
                    y > (-0.1 * Math.pow(x * 10, 2) + 0.6 * Math.sin(4 * x * 10) + 8.0) / 10
                            && y < (2 * Math.sin(x * 10) + 5) / 10.0
                    ||
                    y > ((Math.pow(x * 10 - 10, 2)) / 2 + 7.902) / 10.0) {
                label = 1.0;
            }else{
                label = 0.0;
            }
            // checks whether the instance belongs to the desired class
            if(balanceClassesOption.isSet()) {
                if ((label == 1.0 && nextInstanceShouldBeOne) || (label == 0.0 && !nextInstanceShouldBeOne)) {
                    desiredClassFound = true;
                }
            }else{
                desiredClassFound = true;
            }
        }

        // switches the flag
        nextInstanceShouldBeOne = !nextInstanceShouldBeOne;

        // construct instance
        InstancesHeader header = getHeader();
        Instance inst = new DenseInstance(header.numAttributes());
        inst.setValue(0, x);
        inst.setValue(1, y);
        inst.setDataset(header);
        inst.setClassValue(label);
        return new InstanceExample(inst);
    }

    @Override
    public void restart() {
        this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
        this.nextInstanceShouldBeOne = false;
    }

    @Override
    public InstancesHeader getHeader() {
        return streamHeader;
    }

    @Override
    public long estimatedRemainingInstances() {
        return -1;
    }

    @Override
    public boolean hasMoreInstances() {
        return true;
    }


    @Override
    public boolean isRestartable() {
        return true;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {}
}
