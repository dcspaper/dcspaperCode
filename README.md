## Double Dynamic Classifier Selection

The class of the classifier is in ```moa_code/moa/src/main/java/moa/classifiers/meta/ManualBagThread.java```. 

It has to be executed with task ```moa_code/moa/src/main/java/moa/tasks/EvaluateInterleavedChunksManualBag.java```, which is a copy of the original ```EvaluateInterleavedChunks```, that passes the chunk at once to the classifier.

The built jar file is under the name of ```moa-pom.jar```, and the python script that performed the experiments are under the name of ```moa_caller.py```.

The whole set of results obtained are in the directory ```results```.