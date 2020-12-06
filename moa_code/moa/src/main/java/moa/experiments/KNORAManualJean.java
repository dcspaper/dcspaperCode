package moa.experiments;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.functions.Perceptron;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.classifiers.meta.AdaptiveRandomForest;
import moa.classifiers.meta.OzaBag;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.Utils;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.streams.ImbalancedStream;
import moa.streams.generators.AgrawalGenerator;
import moa.streams.generators.SEAGenerator;
import moa.streams.generators.ValentiniP2;

import java.util.*;
import java.util.stream.Collectors;

public class KNORAManualJean {

    public static void main(String[] args) throws Exception {

        // Instancia ARF
        AbstractClassifier ensemble = new AdaptiveRandomForest();
        ((AdaptiveRandomForest) ensemble).treeLearnerOption.
                setValueViaCLIString("ARFHoeffdingTree -e 2000000 -g 50 -c 0.01 -T 5");
        //setValueViaCLIString("ARFHoeffdingTree -e 2000000 -g 500 -c 0.01 -l MC -T 4");
        ((AdaptiveRandomForest) ensemble).numberOfJobsOption.setValue(4);
        ((AdaptiveRandomForest) ensemble).lambdaOption.setValue(1);
        ((AdaptiveRandomForest) ensemble).disableWeightedVote.set();
        ((AdaptiveRandomForest) ensemble).ensembleSizeOption.setValue(10);
        ensemble.prepareForUse();
        ensemble.resetLearning();

        // Instancia Online Bagging
//        AbstractClassifier ensemble = new OzaBag();
//        ((OzaBag) ensemble).baseLearnerOption.setCurrentObject(new Perceptron());
//        ((OzaBag) ensemble).ensembleSizeOption.setValue(100);
//        ensemble.prepareForUse();
//        ensemble.resetLearning();

        // Instancia stream
//        AgrawalGenerator stream = new AgrawalGenerator();
//        stream.balanceClassesOption.set();
//        stream.prepareForUse();
//        stream.restart();
//

        ValentiniP2 baseStream = new ValentiniP2();
        baseStream.balanceClassesOption.set();
        baseStream.prepareForUse();
        baseStream.restart();

        ImbalancedStream stream = new ImbalancedStream();
        stream.classRatioOption.setValue("0.9;0.1");
        stream.streamOption.setCurrentObject(baseStream);
        stream.prepareForUse();
        stream.restart();


        // 500k pra treino
        LinkedList<Instance> treino = new LinkedList<>();
        while(treino.size() != 10000){
            treino.add(stream.nextInstance().getData());
//            if(treino.size() % 1000 == 0) System.out.println("\t" + treino.size());
        }

        // 2k pra validacao
        LinkedList<Instance> validacao = new LinkedList<>();
        Instances validationSet = new Instances(stream.getHeader());
        while(validacao.size() != 500){
            Instance inst = stream.nextInstance().getData();
            validacao.add(inst);
            validationSet.add(inst);
//            if(validacao.size() % 1000 == 0) System.out.println("\t" + validacao.size());
        }
        validationSet.setClassIndex(validationSet.numAttributes() - 1);

        // 100k pra teste
        LinkedList<Instance> teste = new LinkedList<>();
        while(teste.size() != 100000){
            teste.add(stream.nextInstance().getData());
//            if(teste.size() % 1000 == 0) System.out.println("\t" + teste.size());
        }

        // Treinar o ARF
        System.out.println(" TRAINING ");
        long instancesSeen = 0;
        for (Instance instnc : treino){
            ensemble.trainOnInstance(instnc);
            instancesSeen++;
            if(instancesSeen % 1000 == 0) System.out.println("\t" + instancesSeen);
        }

        for(Measurement m : ensemble.getModelMeasurements()){
            System.out.println(m.getName() + "\t" + m.getValue());
        }

        System.out.println(" VALIDATION ");
        // Using the validation set:
        // - Store a set for  each instance with all classifiers the correctly predicted it
        LinkedList<LinkedList<Classifier>> hits = new LinkedList<>();
        LinkedList<LinkedList<AdaptiveRandomForest.ARFBaseLearner>> hitsARF = new LinkedList<>();
        if(ensemble instanceof AdaptiveRandomForest){
            instancesSeen = 0;
            for (Instance val : validacao){
                LinkedList<AdaptiveRandomForest.ARFBaseLearner> learners = new LinkedList<>();
                for(AdaptiveRandomForest.ARFBaseLearner tree : ((AdaptiveRandomForest) ensemble).ensemble){
                    double[] v = tree.classifier.getVotesForInstance(val);
                    if(Utils.sum(v) > 0.0 && Utils.maxIndex(v) == (int) val.classValue()){
                        learners.add(tree);
                    }
                }

                instancesSeen++;
                if(instancesSeen % 1000 == 0) System.out.println("\t" + instancesSeen);

                // stores in the hash
                hitsARF.add(learners);
            }
        }else if(ensemble instanceof OzaBag){
            for (Instance val : validacao){
                LinkedList<Classifier> learners = new LinkedList<>();
                for(Classifier learner : ((OzaBag) ensemble).ensemble){
                    if(learner.correctlyClassifies(val)){
                        learners.add(learner);
                    }
                }

                // stores in the hash
                hits.add(learners);
            }
        }

        System.out.println(" TEST ");

        // Evaluators
        BasicClassificationPerformanceEvaluator evalTraditional = new BasicClassificationPerformanceEvaluator();
        evalTraditional.precisionRecallOutputOption.set();
        evalTraditional.precisionPerClassOption.set();
        evalTraditional.recallPerClassOption.set();
        BasicClassificationPerformanceEvaluator evalKNORAE = new BasicClassificationPerformanceEvaluator();
        evalKNORAE.precisionRecallOutputOption.set();
        evalKNORAE.precisionPerClassOption.set();
        evalKNORAE.recallPerClassOption.set();
        BasicClassificationPerformanceEvaluator evalKNORAU = new BasicClassificationPerformanceEvaluator();
        evalKNORAU.precisionRecallOutputOption.set();
        evalKNORAU.precisionPerClassOption.set();
        evalKNORAU.recallPerClassOption.set();
        BasicClassificationPerformanceEvaluator evalKNORAEWeighted = new BasicClassificationPerformanceEvaluator();
        BasicClassificationPerformanceEvaluator evalKNORAUWeighted = new BasicClassificationPerformanceEvaluator();

        // Implementa KNORA-E e -U
        instancesSeen = 0;
        for(Instance instnc : teste){
            double[] arfVotes = ensemble.getVotesForInstance(instnc);
            double[] knoraEVotes;
            double[] knoraUVotes;
            double[] knoraEVotesWeighted;
            double[] knoraUVotesWeighted;
            if(ensemble instanceof AdaptiveRandomForest) {
                knoraEVotes = getKNORAE(((AdaptiveRandomForest) ensemble), hitsARF, validacao, validationSet, instnc, false);
                knoraUVotes = getKNORAU(((AdaptiveRandomForest) ensemble), hitsARF, validacao, validationSet, instnc, false);
                knoraEVotesWeighted = getKNORAE(((AdaptiveRandomForest) ensemble), hitsARF, validacao, validationSet, instnc, true);
                knoraUVotesWeighted = getKNORAU(((AdaptiveRandomForest) ensemble), hitsARF, validacao, validationSet, instnc, true);
            }else {
                knoraEVotes = getKNORAE(ensemble, hits, validacao, validationSet, instnc, false);
                knoraUVotes = getKNORAU(ensemble, hits, validacao, validationSet, instnc, false);
                knoraEVotesWeighted = getKNORAE(ensemble, hits, validacao, validationSet, instnc, true);
                knoraUVotesWeighted = getKNORAU(ensemble, hits, validacao, validationSet, instnc, true);
            }

            // updates all evals
            InstanceExample instncExample = new InstanceExample(instnc);
            evalTraditional.addResult(instncExample, arfVotes);
            evalKNORAE.addResult(instncExample, knoraEVotes);
            evalKNORAU.addResult(instncExample, knoraUVotes);
            evalKNORAEWeighted.addResult(instncExample, knoraEVotesWeighted);
            evalKNORAUWeighted.addResult(instncExample, knoraUVotesWeighted);

            instancesSeen++;
            if(instancesSeen % 1000 == 0) System.out.println("\t" + instancesSeen);
        }

        // Output
        System.out.println("TRADITIONAL");
        System.out.println(evalTraditional.toString());
        System.out.println("KNORA E");
        System.out.println(evalKNORAE.toString());
        System.out.println("KNORA U");
        System.out.println(evalKNORAU.toString());
        System.out.println("KNORA E WEIGHTED");
        System.out.println(evalKNORAEWeighted.toString());
        System.out.println("KNORA U WEIGHTED");
        System.out.println(evalKNORAUWeighted.toString());
    }

    private static double[] getKNORAE(AdaptiveRandomForest arf,
                                      LinkedList<LinkedList<AdaptiveRandomForest.ARFBaseLearner>> hits,
                                      LinkedList<Instance> validation,
                                      Instances validationSet,
                                      Instance instnc,
                                      boolean weighted) throws Exception {
        NearestNeighbourSearch nnSearch = new KDTree();//new LinearNNSearch();
        nnSearch.setInstances(validationSet);
        Instances neighbours = nnSearch.kNearestNeighbours(instnc, 5);

        LinkedList<AdaptiveRandomForest.ARFBaseLearner> intersection = new LinkedList<>();
        for(int i = 0; i < neighbours.numInstances(); i++){
            Instance neighbour = neighbours.get(i);
            long ix = validation.indexOf(neighbour);
            LinkedList<AdaptiveRandomForest.ARFBaseLearner> goodPredictors = hits.get((int) ix);

            if (goodPredictors != null && goodPredictors.size() > 0) {
                if (i == 0) {
                    intersection.addAll(goodPredictors);
                } else {
                    Set<AdaptiveRandomForest.ARFBaseLearner> newIntersection = goodPredictors.stream()
                            .distinct()
                            .filter(intersection::contains)
                            .collect(Collectors.toSet());
                    intersection = new LinkedList<>(newIntersection);
                }
            }
        }

        // Given that we have an intersection, let's do a majority vote
        // with the resulting good classifiers
        double[] finalVotes = new double[instnc.numClasses()];
        for (AdaptiveRandomForest.ARFBaseLearner l : intersection){
            double[] v = l.getVotesForInstance(instnc);
            long ix = Utils.maxIndex(v);
            double weight = weighted ? l.evaluator.getPerformanceMeasurements()[1].getValue() : 1;
            finalVotes[(int) ix] += weight;
        }
        return finalVotes;
    }

    private static double[] getKNORAU(AdaptiveRandomForest arf,
                                      LinkedList<LinkedList<AdaptiveRandomForest.ARFBaseLearner>> hits,
                                      LinkedList<Instance> validation,
                                      Instances validationSet,
                                      Instance instnc,
                                      boolean weighted) throws Exception {
        NearestNeighbourSearch nnSearch = new LinearNNSearch();
        nnSearch.setInstances(validationSet);
        Instances neighbours = nnSearch.kNearestNeighbours(instnc, 5);

        LinkedList<AdaptiveRandomForest.ARFBaseLearner> all = new LinkedList<>();
        for(int i = 0; i < neighbours.numInstances(); i++){
            Instance neighbour = neighbours.get(i);
            long ix = validation.indexOf(neighbour);
            LinkedList<AdaptiveRandomForest.ARFBaseLearner> goodPredictors = hits.get((int) ix);

            if (goodPredictors != null && goodPredictors.size() > 0) {
                all.addAll(goodPredictors);
            }
        }

        Set<AdaptiveRandomForest.ARFBaseLearner> union = new LinkedHashSet<>();
        union.addAll(all);

        // Given that we have an intersection, let's do a majority vote
        // with the resulting good classifiers
        double[] finalVotes = new double[instnc.numClasses()];
        for (AdaptiveRandomForest.ARFBaseLearner l : union){
            double[] v = l.getVotesForInstance(instnc);
            long ix = Utils.maxIndex(v);
            double weight = weighted ? l.evaluator.getPerformanceMeasurements()[1].getValue() : 1;
            finalVotes[(int) ix] += weight;
        }
        return finalVotes;

    }

    private static double[] getKNORAE(AbstractClassifier ensemble,
                                      LinkedList<LinkedList<Classifier>> hits,
                                      LinkedList<Instance> validation,
                                      Instances validationSet,
                                      Instance instnc,
                                      boolean weighted) throws Exception {
        NearestNeighbourSearch nnSearch = new LinearNNSearch();
        nnSearch.setInstances(validationSet);
        Instances neighbours = nnSearch.kNearestNeighbours(instnc, 5);

        LinkedList<Classifier> intersection = new LinkedList<>();
        for(int i = 0; i < neighbours.numInstances(); i++){
            Instance neighbour = neighbours.get(i);
            long ix = validation.indexOf(neighbour);
            LinkedList<Classifier> goodPredictors = hits.get((int) ix);

            if (goodPredictors != null && goodPredictors.size() > 0) {
                if (i == 0) {
                    intersection.addAll(goodPredictors);
                } else {
                    Set<Classifier> newIntersection = goodPredictors.stream()
                            .distinct()
                            .filter(intersection::contains)
                            .collect(Collectors.toSet());
                    intersection = new LinkedList<>(newIntersection);
                }
            }
        }

        // Given that we have an intersection, let's do a majority vote
        // with the resulting good classifiers
        double[] finalVotes = new double[instnc.numClasses()];
        for (Classifier l : intersection){
            double[] v = l.getVotesForInstance(instnc);
            long ix = Utils.maxIndex(v);
            //double weight = weighted ? l.evaluator.getPerformanceMeasurements()[1].getValue() : 1;
            finalVotes[(int) ix] += 1;
        }
        return finalVotes;
    }

    private static double[] getKNORAU(AbstractClassifier ensemble,
                                      LinkedList<LinkedList<Classifier>> hits,
                                      LinkedList<Instance> validation,
                                      Instances validationSet,
                                      Instance instnc,
                                      boolean weighted) throws Exception {
        NearestNeighbourSearch nnSearch = new LinearNNSearch();
        nnSearch.setInstances(validationSet);
        Instances neighbours = nnSearch.kNearestNeighbours(instnc, 5);

        LinkedList<Classifier> all = new LinkedList<>();
        for(int i = 0; i < neighbours.numInstances(); i++){
            Instance neighbour = neighbours.get(i);
            long ix = validation.indexOf(neighbour);
            LinkedList<Classifier> goodPredictors = hits.get((int) ix);

            if (goodPredictors != null && goodPredictors.size() > 0) {
                all.addAll(goodPredictors);
            }
        }

        Set<Classifier> union = new LinkedHashSet<>();
        union.addAll(all);

        // Given that we have an intersection, let's do a majority vote
        // with the resulting good classifiers
        double[] finalVotes = new double[instnc.numClasses()];
        for (Classifier l : union){
            double[] v = l.getVotesForInstance(instnc);
            long ix = Utils.maxIndex(v);
            //double weight = weighted ? l.evaluator.getPerformanceMeasurements()[1].getValue() : 1;
            finalVotes[(int) ix] += 1;
        }
        return finalVotes;

    }
}
