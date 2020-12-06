package moa.classifiers.meta;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;

import java.util.LinkedList;

public class ARFDCS extends AbstractClassifier implements MultiClassClassifier {

    /**
     * Options
     */
    public ClassOption arfOption = new ClassOption("arf", 'a', "",
            AdaptiveRandomForest.class,
            "AdaptiveRandomForest");

    public FloatOption validationProbabilityOption = new FloatOption("validationProbability",
            'v',
            "Probability of each instance being used for validation instead of training",
            0.1, 0.0, 1.0);

    public IntOption maxValidationSizeOption = new IntOption("maxValidationSize", 'm', "",
            200, 1, 10000);

    public MultiChoiceOption selectionOption = new MultiChoiceOption("selection", 's', "",
            new String[]{"DEFAULT", "ORACLE", "META-LEARNING"}, new String[]{"DEFAULT", "ORACLE", "META-LEARNING"}, 0);

    public ClassOption dcsLearnerOption = new ClassOption("dcsLearner", 'l', "",
            Classifier.class,
            "bayes.NaiveBayes");

    /**
     * Internals
     */
    AdaptiveRandomForest arf = null;
    Instances validation = null;
    MetaClassifierDCS metaClassifierDCS = null;

    @Override
    public double[] getVotesForInstance(Instance inst) {
        // Prepares the structures
        if(arf == null) prepare(inst);

        // Picks the best performing classifier from ARF for that specific instance
        switch(selectionOption.getChosenLabel()){
            case "ORACLE":
                return selectOracle(inst);
            case "META-LEARNING":
                return selectMetaLearning(inst);
            default:
                return selectDefault(inst);
        }
    }

    @Override
    public void resetLearningImpl() {
        arf = null;
        validation = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        // Prepares the structures
        if(arf == null) prepare(inst);

        float p = classifierRandom.nextFloat();
        if (p < validationProbabilityOption.getValue()){
            // Validation set
            validation.add(inst);

            // MetaHT
            if (this.selectionOption.getChosenLabel().equals("HT")){
                for (int i = 0; i < this.arf.ensemble.length; i++){
                    AdaptiveRandomForest.ARFBaseLearner tree = arf.ensemble[i];
                    double[] treeVotes = tree.getVotesForInstance(inst);
                    if(Utils.max(treeVotes) == (int) inst.classValue()){
                        metaClassifierDCS.update(inst, i);
                    }
                }
            }

            // Drops older instances
            if (validation.size() > maxValidationSizeOption.getValue()){
                validation.delete(0);
            }
        }else{
            // Training
            arf.trainOnInstance(inst);
        }

    }

    private void prepare(Instance instnc) {
        this.arf = (AdaptiveRandomForest) getPreparedClassOption(arfOption);
        this.arf.prepareForUse();
        this.arf.resetLearning();
        this.validation = new Instances(instnc.dataset());
        if(selectionOption.getChosenLabel().equals("HT")){
            metaClassifierDCS = new MetaClassifierDCS(instnc, arf.ensembleSizeOption.getValue(), dcsLearnerOption);
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        out.append("A classifier selector for ARF");
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    /**
     * Returns the oracle for a test instance.
     * @param testInstnc
     * @return
     */
    public double[] selectOracle(Instance testInstnc){
        if(arf.ensemble != null) {
            for (AdaptiveRandomForest.ARFBaseLearner tree : arf.ensemble) {
                double[] votes = tree.getVotesForInstance(testInstnc);
                if (Utils.maxIndex(votes) == testInstnc.classValue()) return votes;
            }
        }
        // let's make sure we get this wrong
        double votes[] = new double[testInstnc.numClasses()];
        int ix = testInstnc.classValue() > 0 ? (int) (testInstnc.classValue() - 1) : 0;
        votes[ix] = 1.0;
        return votes;
    }

    /**
     * Traditional voting scheme using dynamic weighted majority based on accuracy.
     * @param testInstnc
     * @return
     */
    public double[] selectDefault(Instance testInstnc){
        return arf.getVotesForInstance(testInstnc);
    }

    private double[] selectMetaLearning(Instance inst) {
        // determines what is the most appropriate classifier in the ensemble
        if(arf.ensemble != null) {
            int ix = metaClassifierDCS.predict(inst);
            return arf.ensemble[ix].getVotesForInstance(inst);
        }
        return new double[inst.numClasses()];
    }

    class MetaClassifierDCS {
        Classifier learner;
        InstancesHeader header;

        public MetaClassifierDCS(Instance origInstnc, int ensembleSize, ClassOption dcsLearnerOption){
            this.header = new InstancesHeader(origInstnc.dataset());
            LinkedList<String> classVals = new LinkedList<>();
            while (classVals.size() != ensembleSize) classVals.add(String.valueOf((classVals.size() + 1)));
            Attribute newClass = new Attribute("learner", classVals);
            this.header.insertAttributeAt(newClass, this.header.numAttributes());
            this.header.setClassIndex(this.header.numAttributes() - 1);

            this.learner = (Classifier) getPreparedClassOption(dcsLearnerOption);
            this.learner.prepareForUse();
            this.learner.resetLearning();
        }

        public void update(Instance inst, int i) {
            Instance metaInstnc = createInstance(inst, i);

            // Updates the model with the index of the good classifier
            this.learner.trainOnInstance(metaInstnc);
        }

        private Instance createInstance(Instance inst, int i) {
            Instance metaInstnc = new DenseInstance(this.header.numAttributes());
            metaInstnc.setDataset(this.header);

            // data copy
            for(int ix = 0; ix < inst.numAttributes(); ix++){
                metaInstnc.setValue(ix, inst.value(ix));
            }

            // sets the 'label', which is the learner
            metaInstnc.setClassValue(i);
            return metaInstnc;
        }

        public int predict(Instance inst) {
            Instance metaInstnc = createInstance(inst, Integer.MIN_VALUE);
            double[] votes = this.learner.getVotesForInstance(metaInstnc);
            return Utils.maxIndex(votes);
        }
    }

}
