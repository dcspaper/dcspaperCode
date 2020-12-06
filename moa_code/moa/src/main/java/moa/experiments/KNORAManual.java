/*
package moa.experiments;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.classifiers.meta.AdaptiveRandomForest;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.streams.generators.AgrawalGenerator;
import moa.streams.generators.SEAGenerator;
import moa.streams.generators.ValentiniP2;

import java.util.*;
import java.util.stream.Collectors;

class ARFTest extends AdaptiveRandomForest {
    private static final int numNeighbours = 7;
    private NearestNeighbourSearch searchKDTree = new KDTree();

    public void createNeighbours(Instances validacao) {
        try {
            searchKDTree.setInstances(validacao);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Instances findKNN(Instance instancia) {
        Instances neighbours = null;

        try {
            neighbours = searchKDTree.kNearestNeighbours(instancia, numNeighbours);
        } catch (Exception ignored) {
            System.out.println("Error searching for nearest neighbours...");
        }

        return neighbours;
    }

    public int knoraU(Instance instancia) {
        Instances neighbours = findKNN(instancia);

        if (neighbours == null) {
            return 0;
        }

        int[] result = new int[instancia.numClasses()];

        for (int i = 0; i < this.ensemble.length; ++i) {
            double[] votes = this.ensemble[i].getVotesForInstance(instancia);
            DoubleVector voteVector = new DoubleVector(votes);
            int actualPredictedClass = voteVector.maxIndex();

            int numVoteRep = 0;



            for (int j = 0; j < neighbours.size(); j++) {
                Instance neighbour = neighbours.get(j);
                int actualNeighClass = (int) neighbour.classValue();

                double[] votesNeigh = this.ensemble[i].getVotesForInstance(neighbour);
                DoubleVector voteVectorNeigh = new DoubleVector(votesNeigh);
                int predictedNeighClass = voteVectorNeigh.maxIndex();

                if (predictedNeighClass == actualNeighClass) {
                    result[actualPredictedClass]++;
                    numVoteRep++;
                }
            }

            if (numVoteRep > 0) {
                System.out.println("Learner = " + i + " | Predicted Class = " +  actualPredictedClass + " x " + numVoteRep + " times");
            }
        }

        int votoFinal = 0;
        int maisVotos = 0;

        for (int i = 0, resultLength = result.length; i < resultLength; i++) {
            int votos = result[i];

            if (votos > maisVotos) {
                maisVotos = votos;
                votoFinal = i;
            }
        }

        return votoFinal;
    }

    public int knoraE(Instance instancia) {
        Instances neighbours = findKNN(instancia);

        if (neighbours == null) {
            return 0;
        }

        int voteThreshold = neighbours.size();
        int countCorrectVotes = 0;
        int[] result = new int[instancia.numClasses()];
        boolean foundVote = false;

        for (int i = 0; i < this.ensemble.length; ++i) {
            for (int j = 0; j < neighbours.size(); j++) {
                Instance neighbour = neighbours.get(j);
                int actualClass = (int) neighbour.classValue();

                double[] votes = this.ensemble[i].getVotesForInstance(neighbour);
                DoubleVector voteVector = new DoubleVector(votes);
                int actualPredictedClass = voteVector.maxIndex();

                if (actualPredictedClass == actualClass) {
                    countCorrectVotes++;
                }
            }

            if (countCorrectVotes == voteThreshold) {
                double[] votes = this.ensemble[i].getVotesForInstance(instancia);
                DoubleVector voteVector = new DoubleVector(votes);
                int actualPredictedClass = voteVector.maxIndex();

                result[actualPredictedClass]++;
                foundVote = true;

                System.out.println("Learner = " + i + " | Predicted class = " +  actualPredictedClass);
            }

            if (i == ensemble.length - 1 && !foundVote) {
                i = 0;
                voteThreshold--;
            }

            countCorrectVotes = 0;
        }

        int votoFinal = 0;
        int maisVotos = 0;

        for (int i = 0, resultLength = result.length; i < resultLength; i++) {
            int votos = result[i];

            if (votos > maisVotos) {
                maisVotos = votos;
                votoFinal = i;
            }
        }

        return votoFinal;
    }

    ARFBaseLearner[] getEnsemble() {
        return this.ensemble;
    }
}

public class KNORAManual {
    public static void main(String[] args) {
        // Instancia ARF
        ARFTest arf = new ARFTest();
        arf.prepareForUse();
        arf.resetLearning();

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
        }

        // 2k pra validacao
        Instances validacao = null;

        do {
            InstanceExample aux = stream.nextInstance();

            if (validacao == null) {
                validacao= new Instances(aux.getData().dataset());
            }

            validacao.add(aux.getData());
        } while(validacao.size() != 2000);

        // 1kk pra teste
        LinkedList<Instance> teste = new LinkedList<>();
        int numTeste = 1000000;

        while(teste.size() != numTeste){
            teste.add(stream.nextInstance().getData());
        }

        // Treinar o ARF
        System.out.println(" TRAINING ");
        long instancesSeen = 0;
        for (Instance instnc : treino){
            ensemble.trainOnInstance(instnc);
            instancesSeen++;
            if(instancesSeen % 1000 == 0) System.out.println("\t" + instancesSeen);
        }

        System.out.println("Starting experiment with " + numTeste + " test instances\n");

        arf.createNeighbours(validacao);

        // Implementa KNORA-E e -U
        int acertosKnoraE = 0;
        int acertosKnoraU = 0;
        int acertosARFNormal = 0;

        for (Instance instancia : teste) {
            System.out.println("-----------------------------------------------------");
            System.out.println("Instance real class = " + (int) instancia.classValue());

            System.out.println("\nKnora-E:");
            int votoKnoraE = arf.knoraE(instancia);

            System.out.println("\nKnora-U:");
            int votoKnoraU = arf.knoraU(instancia);

            double[] votosARF = arf.getVotesForInstance(instancia);
            DoubleVector votos = new DoubleVector(votosARF);
            int votoARF = votos.maxIndex();

            int classeReal = (int) instancia.classValue();

            if (votoKnoraE == classeReal) {
                acertosKnoraE++;
            }

            if (votoKnoraU == classeReal) {
                acertosKnoraU++;
            }

            if (votoARF == classeReal) {
                acertosARFNormal++;
            }

            System.out.println("");
        }

        // Acurácia do normal, Knora-E e -U
        double accKnoraE = ((double) acertosKnoraE / (double) numTeste) * 100D;
        double accKnoraU = ((double) acertosKnoraU / (double) numTeste) * 100D;
        double accARF = ((double) acertosARFNormal / (double) numTeste) * 100D;

        System.out.println("Acurácia Knora E = " + accKnoraE + "%");
        System.out.println("Acurácia Knora U = " + accKnoraU + "%");
        System.out.println("Acurácia ARF     = " + accARF + "%");
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
}*/
