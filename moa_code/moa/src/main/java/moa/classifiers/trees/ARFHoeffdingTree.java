/*
 *    ARFHoeffdingTree.java
 * 
 *    @author Heitor Murilo Gomes (heitor_murilo_gomes at yahoo dot com dot br)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */

package moa.classifiers.trees;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.core.Utils;
import com.yahoo.labs.samoa.instances.Instance;

/**
 * Adaptive Random Forest Hoeffding Tree.
 * 
 * <p>Adaptive Random Forest Hoeffding Tree. This is the base model for the 
 * Adaptive Random Forest ensemble learner 
 * (See moa.classifiers.meta.AdaptiveRandomForest.java). This Hoeffding Tree
 * includes a subspace size k parameter, which defines the number of randomly 
 * selected features to be considered at each split. </p>
 * 
 * <p>See details in:<br> Heitor Murilo Gomes, Albert Bifet, Jesse Read, 
 * Jean Paul Barddal, Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, 
 * Talel Abdessalem. Adaptive random forests for evolving data stream classification. 
 * In Machine Learning, DOI: 10.1007/s10994-017-5642-8, Springer, 2017.</p>
 *
 * @author Heitor Murilo Gomes (heitor_murilo_gomes at yahoo dot com dot br)
 * @version $Revision: 1 $
 */
public class ARFHoeffdingTree extends HoeffdingTree {

    private static final long serialVersionUID = 1L;
    
    public IntOption subspaceSizeOption = new IntOption("subspaceSizeSize", 'k',
            "Number of features per subset for each node split. Negative values = #features - k", 
            2, Integer.MIN_VALUE, Integer.MAX_VALUE);

    public FlagOption decaySubspaceOption = new FlagOption("decaySubspace",
            'D', "Apply decaying subspaces according to the tree level.");

    public FlagOption decayGracePeriodOption = new FlagOption("decayGracePeriod", 'G',
            "Decay grace period for split checks.");
    
    @Override
    public String getPurposeString() {
        return "Adaptive Random Forest Hoeffding Tree for data streams. "
                + "Base learner for AdaptiveRandomForest.";
    }

    private int getGracePeriod(int depth, int treeGracePeriod){
        int num = (int) Math.ceil(Math.pow(2, -0.25 * depth) * treeGracePeriod);
        if (num == 0) num = 1;
        return num;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.treeRoot == null) {
            this.treeRoot = newLearningNode(0);
            this.activeLeafNodeCount = 1;
        }
        FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
        Node leafNode = foundNode.node;
        if (leafNode == null) {
            leafNode = newLearningNode(foundNode.parent.depth + 1);
            foundNode.parent.setChild(foundNode.parentBranch, leafNode);
            this.activeLeafNodeCount++;
        }
        if (leafNode instanceof LearningNode) {
            LearningNode learningNode = (LearningNode) leafNode;
            learningNode.learnFromInstance(inst, this);
            if (this.growthAllowed
                    && (learningNode instanceof ActiveLearningNode)) {
                ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
                double weightSeen = activeLearningNode.getWeightSeen();

                // corrected grace period
                int treeGracePeriod = this.gracePeriodOption.getValue();
                int correctedGracePeriod = decayGracePeriodOption.isSet() ? getGracePeriod(learningNode.depth, treeGracePeriod) : treeGracePeriod;

                if (weightSeen
                        - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= correctedGracePeriod) {
                    attemptToSplit(activeLearningNode, foundNode.parent,
                            foundNode.parentBranch);
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                }
            }
        }
        if (this.trainingWeightSeenByModel
                % this.memoryEstimatePeriodOption.getValue() == 0) {
            estimateModelByteSizes();
        }
    }

    public static class RandomLearningNode extends ActiveLearningNode {
        
        private static final long serialVersionUID = 1L;

        protected int[] listAttributes;

        protected int numAttributes;

        boolean useDecay;

//        boolean useGracePeriodDecay;
//        int treeGracePeriod;
        
        public RandomLearningNode(double[] initialClassObservations, int subspaceSize, int depth,
                                  boolean useDecay, boolean useGracePeriodDecay, int treeGracePeriod
                                  ) {
            super(initialClassObservations, depth);
            this.numAttributes = subspaceSize;
            this.useDecay = useDecay;
//            this.useGracePeriodDecay = useGracePeriodDecay;
//            this.treeGracePeriod = treeGracePeriod;
        }

        private int getSubspaceSize(int depth, int numFeatures){
            int num = (int) (Math.ceil(Math.pow(2, -0.25* depth) * numFeatures));
            if (num == 0){
                num = 1;
            }
            return num;
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {            
            this.observedClassDistribution.addToValue((int) inst.classValue(),
                    inst.weight());
            if(this.useDecay) {
                this.numAttributes = getSubspaceSize(depth, inst.numAttributes() - 1);
            }
            if (this.listAttributes == null) {
                this.listAttributes = new int[this.numAttributes];
                for (int j = 0; j < this.numAttributes; j++) {
                    boolean isUnique = false;
                    while (isUnique == false) {
                        this.listAttributes[j] = ht.classifierRandom.nextInt(inst.numAttributes() - 1);
                        isUnique = true;
                        for (int i = 0; i < j; i++) {
                            if (this.listAttributes[j] == this.listAttributes[i]) {
                                isUnique = false;
                                break;
                            }
                        }
                    }

                }
            }
            for (int j = 0; j < this.numAttributes - 1; j++) {
                int i = this.listAttributes[j];
                int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs == null) {
                    obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
                    this.attributeObservers.set(i, obs);
                }
                obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
            }
        }
    }

    public static class LearningNodeNB extends RandomLearningNode {

        private static final long serialVersionUID = 1L;

        public LearningNodeNB(double[] initialClassObservations, int subspaceSize,
                              int depth, boolean useDecay, boolean useGracePeriodDecay, int treeGracePeriod) {
            super(initialClassObservations, subspaceSize, depth, useDecay, useGracePeriodDecay, treeGracePeriod);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            if (getWeightSeen() >= ht.nbThresholdOption.getValue()) {
                return NaiveBayes.doNaiveBayesPrediction(inst,
                        this.observedClassDistribution,
                        this.attributeObservers);
            }
            return super.getClassVotes(inst, ht);
        }

        @Override
        public void disableAttribute(int attIndex) {
            // should not disable poor atts - they are used in NB calc
        }
    }

    public static class LearningNodeNBAdaptive extends LearningNodeNB {

        private static final long serialVersionUID = 1L;

        protected double mcCorrectWeight = 0.0;

        protected double nbCorrectWeight = 0.0;

        public LearningNodeNBAdaptive(double[] initialClassObservations, int subspaceSize, int depth, boolean useDecay, boolean useGracePeriodDecay, int treeGracePeriod) {
            super(initialClassObservations, subspaceSize, depth, useDecay, useGracePeriodDecay, treeGracePeriod);
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            int trueClass = (int) inst.classValue();
            if (this.observedClassDistribution.maxIndex() == trueClass) {
                this.mcCorrectWeight += inst.weight();
            }
            if (Utils.maxIndex(NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers)) == trueClass) {
                this.nbCorrectWeight += inst.weight();
            }
            super.learnFromInstance(inst, ht);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            if (this.mcCorrectWeight > this.nbCorrectWeight) {
                return this.observedClassDistribution.getArrayCopy();
            }
            return NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers);
        }
    }

    public ARFHoeffdingTree() {
        this.removePoorAttsOption = null;
    }

    @Override
    protected LearningNode newLearningNode(double[] initialClassObservations, int depth) {
        LearningNode ret;
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        boolean useDecay = this.decaySubspaceOption.isSet();
        boolean useGracePeriodDecay = this.decayGracePeriodOption.isSet();
        int treeGracePeriod = this.gracePeriodOption.getValue();
        if (predictionOption == 0) { //MC
            ret = new RandomLearningNode(initialClassObservations, this.subspaceSizeOption.getValue(), depth, useDecay, useGracePeriodDecay, treeGracePeriod);
        } else if (predictionOption == 1) { //NB
            ret = new LearningNodeNB(initialClassObservations, this.subspaceSizeOption.getValue(), depth, useDecay, useGracePeriodDecay, treeGracePeriod);
        } else { //NBAdaptive
            ret = new LearningNodeNBAdaptive(initialClassObservations, this.subspaceSizeOption.getValue(), depth, useDecay, useGracePeriodDecay, treeGracePeriod);
        }
        return ret;
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}
