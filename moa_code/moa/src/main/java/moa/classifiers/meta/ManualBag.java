/*
 *    OzaBag.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
package moa.classifiers.meta;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.*;
import meka.core.A;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;

import moa.classifiers.MultiClassClassifier;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.core.*;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;
import com.github.javacliparser.IntOption;

import javax.swing.plaf.synth.SynthEditorPaneUI;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.stream.IntStream;

/**
 * Incremental on-line bagging of Oza and Russell.
 *
 * <p>Oza and Russell developed online versions of bagging and boosting for
 * Data Streams. They show how the process of sampling bootstrap replicates
 * from training data can be simulated in a data stream context. They observe
 * that the probability that any individual example will be chosen for a
 * replicate tends to a Poisson(1) distribution.</p>
 *
 * <p>[OR] N. Oza and S. Russell. Online bagging and boosting.
 * In Artiﬁcial Intelligence and Statistics 2001, pages 105–112.
 * Morgan Kaufmann, 2001.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classiﬁer to train</li>
 * <li>-s : The number of models in the bag</li> </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class ManualBag extends AbstractClassifier implements MultiClassClassifier,
        CapabilitiesHandler {

    @Override
    public String getPurposeString() {
        return "Incremental on-line bagging of Oza and Russell.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    public FlagOption updateAllEnsembleOption = new FlagOption("updateAllEnsemble", 'a',
            "Wheter to update whole ensemble or only the newest member");

    public FlagOption baggingOption = new FlagOption("bagging", 'b',
            "Wheter to use online bagging when updating or not");

    public FlagOption randomSubspaceOption = new FlagOption("randomSubspace", 'r',
            "Wheter to use random subspace when updating or not");

    public FlagOption initEnsembleOption = new FlagOption("initEnsemble", 'i',
            "Wheter to prestart the ensemble or not");

    public FlagOption preprocessAllOption = new FlagOption("preprocessAll", 'c',
            "Preprocess the whole training set or only...");

    public MultiChoiceOption votingMethodOption = new MultiChoiceOption("votingMethod", 'V',
            "",
            new String[]{"NO_SEL", "KNORAE", "KNORAU"},
            new String[]{"NO_SEL", "KNORAE", "KNORAU"},
            0);

    public MultiChoiceOption preprocessMethodOption = new MultiChoiceOption("preprocess", 'p',
            "",
            new String[]{"NO_PREPROCESS", "SMOTE", "BORDERLINESMOTE", "RANDOMOVERSAMPLER"},
            new String[]{"NO_PREPROCESS", "SMOTE", "BORDERLINESMOTE", "RANDOMOVERSAMPLER"},
            0);

    private ArrayList<Classifier> ensemble;
    private ArrayList<BasicClassificationPerformanceEvaluator> evals;
    private ArrayList<ArrayList<Integer>> invalid_atts;

    private KDTree searcher;



    @Override
    public void resetLearningImpl() {
        this.ensemble = new ArrayList<Classifier>();
        this.evals = new ArrayList<BasicClassificationPerformanceEvaluator>();
        this.invalid_atts = new ArrayList<ArrayList<Integer>>();
        if (this.initEnsembleOption.isSet()) {
            Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
            baseLearner.resetLearning();
            BasicClassificationPerformanceEvaluator ev = new BasicClassificationPerformanceEvaluator();
            for (int i = 0; i < this.ensembleSizeOption.getValue(); i++) {
                this.ensemble.add(baseLearner.copy());
                this.evals.add((BasicClassificationPerformanceEvaluator) ev.copy());
            }
        }
    }

    private int get_worse(){
        int index = 0;
        double worse = 101.0;
        for (int i = 0; i < this.evals.size(); i++){
            double curr_acc = this.evals.get(i).getPerformanceMeasurements()[1].getValue();
            if (curr_acc < worse){
                worse = curr_acc;
                index = i;
            }
        }
        return index;
    }

    private ArrayList<Integer> getNRandom(int up_to, int n){
        ArrayList<Integer> arr = new ArrayList<Integer>();
        for (int i = 0; i < up_to; i++){
            arr.add(i);
        }
        Collections.shuffle(arr);
        ArrayList<Integer> ret = new ArrayList<Integer>();
        for (int i = 0; i < n; i++){
            ret.add(arr.get(i));
        }
        return ret;
    }

    private Instances applyPreprocess(Instances instances, InstancesHeader header){
        to_csv("temp.csv", instances);
        Runtime r = Runtime.getRuntime();
        Process p = null;
        try {
            p = r.exec("python preprocesser.py temp.csv ".concat(this.preprocessMethodOption.getChosenLabel()));

            p.waitFor();
            BufferedReader b = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line = "";

            while ((line = b.readLine()) != null) {
                System.out.println(line);
            }

            b.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return from_csv("_temp.csv", header);
    }

    private Instances from_csv(String filename, InstancesHeader header){

        BufferedReader br = null;
        String line = "";
        ArrayList<String[]> insts = new ArrayList<>();
        try {

            br = new BufferedReader(new FileReader(filename));
            while ((line = br.readLine()) != null) {
                if (line.equals("")){
                    continue;
                }

                // use comma as separator
                insts.add(line.split(","));

//                System.out.println("Country [code= " + country[4] + " , name=" + country[5] + "]");

            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        Instances instances = new Instances(header, insts.size());

        for (int i = 0; i < insts.size(); i++) {
            Instance inst = new DenseInstance(header.numAttributes());
            for (int j = 0; j < insts.get(i).length; j++) {
                if (j == insts.get(i).length - 1) {
                    inst.setDataset(header);
                    inst.setClassValue(Double.parseDouble(insts.get(i)[j].substring(insts.get(i)[j].length() - 1))-1);
                }
                else {
                    inst.setValue(j, Double.parseDouble(insts.get(i)[j]));
                }
            }
            instances.add(inst.copy());
        }
        return instances;
    }

    private void to_csv(String filename, Instances instances){
        try {
            FileWriter myWriter = new FileWriter(filename, false);
            for (int i = 0; i < instances.size(); i++) {
                String ins = instances.get(i).toString();
                myWriter.write(ins.substring(0, ins.length() - 1));
                myWriter.write("\n");

            }
            myWriter.close();
//            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    @Override
    public void trainChunk(Instances instances, InstancesHeader header){
        Instances preproc_instances = null;
        if (!this.preprocessMethodOption.getChosenLabel().equals("NO_PREPROCESS")) {
            preproc_instances = applyPreprocess(instances, header);
        }
        this.searcher = new KDTree();
        try {
            if (this.preprocessMethodOption.getChosenLabel().equals("NO_PREPROCESS")) {
                this.searcher.setInstances(instances);
            } else{
                this.searcher.setInstances(preproc_instances);
                if (this.preprocessAllOption.isSet()){
                    instances = preproc_instances;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        BasicClassificationPerformanceEvaluator ev = new BasicClassificationPerformanceEvaluator();
        baseLearner.resetLearning();
        try {
            if (this.ensemble.size() == this.ensembleSizeOption.getValue()){
                int worst_index = get_worse();
                this.ensemble.remove(worst_index);
                this.evals.remove(worst_index);
            }
            this.ensemble.add(baseLearner.copy());
            this.evals.add((BasicClassificationPerformanceEvaluator) ev.copy());

        } catch (Exception e) {
            System.out.println("first time error");
        }

        if (this.updateAllEnsembleOption.isSet()){
            if (this.baggingOption.isSet()){
                for (int i =0; i< instances.size(); i++) {
                    trainAllBagging(instances.get(i));
                }
            } else{
                if (this.randomSubspaceOption.isSet()){
                    for (int i =0; i< instances.size(); i++) {
//                        trainAllNoBaggingSubspace(instances.get(i));
                    }
                } else {
                    for (int i = 0; i < instances.size(); i++) {
                        trainAllNoBagging(instances.get(i));
                    }
                }
            }
        } else {
            for (int i =0; i< instances.size(); i++)  {
                this.ensemble.get(this.ensemble.size() - 1).trainOnInstance(instances.get(i));
            }
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        for (int i = 0; i < this.ensembleSizeOption.getValue(); i++) {
            int k = MiscUtils.poisson(1.0, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) inst.copy();
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble.get(i).trainOnInstance(weightedInst);
            }
        }
    }

    private void trainAllNoBagging(Instance inst){
        for (int i = 0; i < this.ensemble.size(); i++) {
            this.ensemble.get(i).trainOnInstance(inst.copy());
        }
    }

//    private void trainAllNoBaggingSubspace(Instance inst){
//        Instance copyInst = inst.copy();
//        for (int i = 0; i < this.ensemble.size(); i++) {
//            ArrayList<Integer> invalids = this.invalid_atts.get(i);
//            for (int j=invalids.size()-1; j >= 0; j--){
//                if (invalids.get(j) == copyInst.classIndex()){
//                    continue;
//                }
//                copyInst.deleteAttributeAt(invalids.get(j));
//            }
//            copyInst.set
//            this.ensemble.get(i).trainOnInstance(copyInst);
//        }
//    }

    private void trainAllBagging(Instance inst){
        for (int i = 0; i < this.ensemble.size(); i++) {
            int k = MiscUtils.poisson(1.0, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) inst.copy();
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble.get(i).trainOnInstance(weightedInst);
            }
        }
    }

    private void trainAllBaggingSubspace(Instance inst){
        for (int i = 0; i < this.ensemble.size(); i++) {
            int k = MiscUtils.poisson(1.0, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) inst.copy();
                ArrayList<Integer> invalids = this.invalid_atts.get(i);
                for (int j=invalids.size()-1; j >= 0; j--){
                    weightedInst.deleteAttributeAt(invalids.get(j));
                }
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble.get(i).trainOnInstance(weightedInst);
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.votingMethodOption.getChosenLabel().equals("NO_SEL")){
            return getVotesForInstanceNoSel(inst);
        } else if (this.votingMethodOption.getChosenLabel().equals("KNORAE")){
            return getVotesForInstanceKNORAE(inst);
        } else if (this.votingMethodOption.getChosenLabel().equals("KNORAU")){
            return getVotesForInstanceKNORAU(inst);
        }
        return new double[0];
    }


    private double[] getVotesForInstanceKNORAU(Instance inst){
        Example testInst = new InstanceExample((Instance) inst);
        Instances neighbours;
        try {
            neighbours = this.searcher.kNearestNeighbours(inst, 7);
        } catch (Exception e) {
            System.out.println("erro no knorau");
            return getVotesForInstanceNoSel(inst);
//            e.printStackTrace();
        }
        double[] result = new double[inst.numClasses()];

        for (int i = 0; i < this.ensemble.size(); ++i) {
            double[] votes = this.ensemble.get(i).getVotesForInstance(inst);
            DoubleVector voteVector = new DoubleVector(votes);
            int actualPredictedClass = voteVector.maxIndex();

            DoubleVector vote = new DoubleVector(this.ensemble.get(i).getVotesForInstance(inst));
            this.evals.get(i).addResult(testInst, vote.getArrayCopy());

            //int numVoteRep = 0;

            for (int j = 0; j < neighbours.size(); j++) {
                Instance neighbour = neighbours.get(j);
                int actualNeighClass = (int) neighbour.classValue();

                double[] votesNeigh = this.ensemble.get(i).getVotesForInstance(neighbour);
                DoubleVector voteVectorNeigh = new DoubleVector(votesNeigh);
                int predictedNeighClass = voteVectorNeigh.maxIndex();

                if (predictedNeighClass == actualNeighClass) {
                    result[actualPredictedClass]++;
                    //numVoteRep++;
                }
            }
        }

        return result;
    }

    private double[] getVotesForInstanceKNORAURandomSubspace(Instance inst){
        Instances neighbours;
        try {
            neighbours = this.searcher.kNearestNeighbours(inst, 7);
        } catch (Exception e) {
            System.out.println("erro no knorau");
            return getVotesForInstanceNoSel(inst);
//            e.printStackTrace();
        }
        double[] result = new double[inst.numClasses()];

        for (int i = 0; i < this.ensemble.size(); ++i) {
            Instance CopyInst = inst.copy();
            ArrayList<Integer> invalids = this.invalid_atts.get(i);
            for (int j=invalids.size()-1; j >= 0; j--){
                CopyInst.deleteAttributeAt(invalids.get(j));
            }

            double[] votes = this.ensemble.get(i).getVotesForInstance(CopyInst);
            DoubleVector voteVector = new DoubleVector(votes);
            int actualPredictedClass = voteVector.maxIndex();

            DoubleVector vote = new DoubleVector(this.ensemble.get(i).getVotesForInstance(CopyInst));
            Example testInst = new InstanceExample((Instance) CopyInst);
            this.evals.get(i).addResult(testInst, vote.getArrayCopy());


            for (int j = 0; j < neighbours.size(); j++) {
                Instance CopyNeigh = neighbours.get(j).copy();
                ArrayList<Integer> invalidsNeigh = this.invalid_atts.get(i);
                for (int k=invalids.size()-1; k >= 0; k--){
                    CopyInst.deleteAttributeAt(invalidsNeigh.get(k));
                }
                int actualNeighClass = (int) CopyNeigh.classValue();

                double[] votesNeigh = this.ensemble.get(i).getVotesForInstance(CopyNeigh);
                DoubleVector voteVectorNeigh = new DoubleVector(votesNeigh);
                int predictedNeighClass = voteVectorNeigh.maxIndex();

                if (predictedNeighClass == actualNeighClass) {
                    result[actualPredictedClass]++;
                }
            }
        }

        return result;
    }

    private double[] getVotesForInstanceKNORAE(Instance inst){
        Example testInst = new InstanceExample((Instance) inst);
        Instances neighbours;
        try {
            neighbours = this.searcher.kNearestNeighbours(inst, 7);
        } catch (Exception e) {
            System.out.println("erro no knorae");
            return getVotesForInstanceNoSel(inst);
//            e.printStackTrace();
        }
        double[] result = new double[inst.numClasses()];
        int[] correctVotesNumber = new int[this.ensemble.size()];

        for (int j = 0; j < this.ensemble.size(); ++j) {
            DoubleVector vote = new DoubleVector(this.ensemble.get(j).getVotesForInstance(inst));
            this.evals.get(j).addResult(testInst, vote.getArrayCopy());
        }


        for (int i = 0; i < neighbours.size(); i++) {
            Instance neighbour = neighbours.get(i);
            int actualClass = (int) neighbour.classValue();
            for (int j = 0; j < this.ensemble.size(); ++j) {

                double[] votes = this.ensemble.get(j).getVotesForInstance(neighbour);
                DoubleVector voteVector = new DoubleVector(votes);
                int predictedClass = voteVector.maxIndex();
                if (predictedClass == actualClass) {
                    correctVotesNumber[j] += 1;
                }
            }
        }

        int max = Arrays.stream(correctVotesNumber).max().getAsInt();
        int[] indexes = IntStream.range(0, correctVotesNumber.length).filter(i -> correctVotesNumber[i] == max).toArray();


        for (int i: indexes) {
            double[] votes = this.ensemble.get(i).getVotesForInstance(inst);
            DoubleVector voteVector = new DoubleVector(votes);
            int predictedClass = voteVector.maxIndex();
            if (predictedClass != -1){
                result[predictedClass] += 1;
            }
        }

        return result;
    }

    private double[] getVotesForInstanceNoSel(Instance inst) {
        Example testInst = new InstanceExample((Instance) inst);
        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this.ensemble.size(); i++) {
            DoubleVector vote = new DoubleVector(this.ensemble.get(i).getVotesForInstance(inst));
            this.evals.get(i).addResult(testInst, vote.getArrayCopy());
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size",
                this.ensemble != null ? this.ensembleSizeOption.getValue() : 0)};
    }

    @Override
    public Classifier[] getSubClassifiers() {
        Classifier[] arr = this.ensemble.toArray(new Classifier[this.ensemble.size()]);
        return arr.clone();
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == ManualBag.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

    public Classifier[] getEnsemble() {
        Classifier[] arr = this.ensemble.toArray(new Classifier[this.ensemble.size()]);
        return arr.clone();
    }
}
