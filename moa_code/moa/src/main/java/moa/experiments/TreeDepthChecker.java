package moa.experiments;

import moa.classifiers.meta.AdaptiveRandomForestDCS;
import moa.classifiers.trees.ARFHoeffdingTreeDCS;
import moa.streams.generators.SEAGenerator;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;

public class TreeDepthChecker {

    private static String testARFHoeffdingTreeDCS(int instancesNum) {
        SEAGenerator stream = new SEAGenerator();
        stream.prepareForUse();

        ARFHoeffdingTreeDCS tree = new ARFHoeffdingTreeDCS();
        tree.prepareForUse();

        for (int i = 0; i < instancesNum; i++) tree.trainOnInstance(stream.nextInstance());

        return tree.toString();
    }

    private static String testAdaptiveRandomForestDCS(int instancesNum) {
        SEAGenerator stream = new SEAGenerator();
        stream.prepareForUse();

        AdaptiveRandomForestDCS ensemble = new AdaptiveRandomForestDCS();
        ensemble.prepareForUse();

        for (int i = 0; i < instancesNum; i++) ensemble.trainOnInstance(stream.nextInstance());

        return ensemble.toString();
    }

    private static String testAdaptiveRandomForestOracleDCS(int instancesNum) {
        SEAGenerator stream = new SEAGenerator();
        stream.prepareForUse();

        AdaptiveRandomForestDCS ensemble = new AdaptiveRandomForestDCS();
        ensemble.votingMethodOption.setChosenLabel("Oracle Tree Depth");
        ensemble.prepareForUse();

        for (int i = 0; i < instancesNum; i++) ensemble.trainOnInstance(stream.nextInstance());

        int testInstancesNum = (int) (instancesNum * 0.3);

        System.out.println("Test predictions (" + testInstancesNum+ " instances):");
        for (int i = 0; i < testInstancesNum; i++) ensemble.getVotesForInstance(stream.nextInstance());

        return ensemble.toString();
    }

    public static void main(String args[]) throws FileNotFoundException {
        String description = testAdaptiveRandomForestOracleDCS(500000);

        try (PrintWriter out = new PrintWriter("OracleDCSTreeDepthCheckResults.txt")) {
            out.println(description);
            System.out.println("Results file created!");
        }
    }
}