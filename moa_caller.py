import os, time, random

def reduce_name(name):
    if "ArffFileStream" in name:
        return name.split("ArffFileStream")[1].split(" ")[-1].replace("bases/","").replace(")", "")

    if "SEAGenerator" in name:
        new_name = "SEAGenerator - "
    elif "Agrawal" in name:
        new_name = "Agrawal - "
    elif "asset" in name.lower():
        new_name = "Asset - "
    
    concept_times = name.count("ConceptDriftStream")
    if concept_times > 0:
        if concept_times == 1:
            new_name += "Concept 2X "
        elif concept_times == 2:
            new_name += "Concept 3X "
        elif concept_times == 3:
            new_name += "Concept 4X "
        
        if "-w 0" in name:
            new_name += "Sudden"
        else:
            new_name += "Gradual"
    else:
        new_name += "No drift"
    
    return new_name

def check_artificial(name):
    if "SEAGenerator" in name:
        return True
    elif "Agrawal" in name:
        return True
    elif "asset" in name.lower():
        return True
    return False

def replace_seed(string):
    for i in range(string.count("SEED")):
        string = string.replace("SEED", str(random.randint(0, 10000)), 1)

    return string


# random_state = np.random.RandomState(1)

random.seed(1)

chunk_size = 1000

base_string_proposed = f"java -Xmx8g -javaagent:sizeofag-1.0.0.jar -cp moa-pom.jar moa.DoTask \"EvaluateInterleavedChunksManualBag -l (moa.classifiers.meta.ManualBagThreadUSE_ALLUSE_BAGGINGUSE_INIT -V DCS_METHOD -s SIZE) -s STREAM -i N_INSTANCES -f {chunk_size} -q {chunk_size} -c {chunk_size} -d 'OUTPUT_FILE'\""
base_string_baseline = f"java -Xmx8g -javaagent:sizeofag-1.0.0.jar -cp moa-pom.jar moa.DoTask \"EvaluateInterleavedChunks -l CLF -s STREAM -i N_INSTANCES -f {chunk_size} -q {chunk_size} -c {chunk_size} -d 'OUTPUT_FILE'\""

dcss = [
    "KNORAU",
    "KNORAE",
    "NO_SEL"
]

bags = [True, False]
alls = [True, False]
inits = [True, False]

sizes = [100]

clfs = [
    "(moa.classifiers.meta.AdaptiveRandomForest -o (Specified m (integer value)) -m 2 -j 4 -s 100)",
    "(moa.classifiers.meta.AdaptiveRandomForest -o (Specified m (integer value)) -m 3 -j 4 -s 100)",
    "(moa.classifiers.meta.AdaptiveRandomForest -o (Specified m (integer value)) -m 4 -j 4 -s 100)",
    "(moa.classifiers.meta.AdaptiveRandomForest -o (Specified m (integer value)) -m 5 -j 4 -s 100)",
    "(moa.classifiers.meta.AdaptiveRandomForest -o (Specified m (integer value)) -m 6 -j 4 -s 100)",
    "(moa.classifiers.meta.AdaptiveRandomForest -j 4 -s 100)",
    "(moa.classifiers.meta.OzaBag -s 100)"
]

streams = [
    "(generators.SEAGenerator -i SEED)",
    "(generators.AgrawalGenerator -i SEED)",
    "(generators.AssetNegotiationGenerator -i SEED)",

    "(ConceptDriftStream -s (generators.SEAGenerator -i SEED) -d (generators.SEAGenerator -i SEED -f 2) -p 50000 -w 0)",
    "(ConceptDriftStream -s (generators.SEAGenerator -i SEED) -d (generators.SEAGenerator -i SEED -f 2) -p 50000 -w 1000)",
    "(ConceptDriftStream -s (generators.AgrawalGenerator -i SEED) -d (generators.AgrawalGenerator -i SEED -f 2) -p 50000 -w 0)",
    "(ConceptDriftStream -s (generators.AgrawalGenerator -i SEED) -d (generators.AgrawalGenerator -i SEED -f 2) -p 50000 -w 1000)",
    "(ConceptDriftStream -s (generators.AssetNegotiationGenerator -i SEED) -d (generators.AssetNegotiationGenerator -i SEED -f 2) -p 50000 -w 0)",
    "(ConceptDriftStream -s (generators.AssetNegotiationGenerator -i SEED) -d (generators.AssetNegotiationGenerator -i SEED -f 2) -p 50000 -w 1000)",
    
    "(ConceptDriftStream -s (generators.SEAGenerator -i SEED) -d (ConceptDriftStream -s (generators.SEAGenerator -i SEED -f 2) -d (generators.SEAGenerator -i SEED -f 3) -p 33300 -w 0) -p 33300 -w 0)",
    "(ConceptDriftStream -s (generators.SEAGenerator -i SEED) -d (ConceptDriftStream -s (generators.SEAGenerator -i SEED -f 2) -d (generators.SEAGenerator -i SEED -f 3) -p 33300 -w 1000) -p 33300 -w 1000)",
    "(ConceptDriftStream -s (generators.AgrawalGenerator -i SEED) -d (ConceptDriftStream -s (generators.AgrawalGenerator -i SEED -f 2) -d (generators.AgrawalGenerator -i SEED -f 3) -p 33300 -w 0) -p 33300 -w 0)",
    "(ConceptDriftStream -s (generators.AgrawalGenerator -i SEED) -d (ConceptDriftStream -s (generators.AgrawalGenerator -i SEED -f 2) -d (generators.AgrawalGenerator -i SEED -f 3) -p 33300 -w 1000) -p 33300 -w 1000)",
    "(ConceptDriftStream -s (generators.AssetNegotiationGenerator -i SEED) -d (ConceptDriftStream -s (generators.AssetNegotiationGenerator -i SEED -f 2) -d (generators.AssetNegotiationGenerator -i SEED -f 3) -p 33300 -w 0) -p 33300 -w 0)",
    "(ConceptDriftStream -s (generators.AssetNegotiationGenerator -i SEED) -d (ConceptDriftStream -s (generators.AssetNegotiationGenerator -i SEED -f 2) -d (generators.AssetNegotiationGenerator -i SEED -f 3) -p 33300 -w 1000) -p 33300 -w 1000)",

    "(ConceptDriftStream -s (ConceptDriftStream -s (generators.SEAGenerator -i SEED) -d (generators.SEAGenerator -i SEED -f 2) -p 25000 -w 0) -d (ConceptDriftStream -s (generators.SEAGenerator -i SEED -f 3) -d (generators.SEAGenerator -i SEED -f 4) -p 25000 -w 0) -p 25000 -w 0)",
    "(ConceptDriftStream -s (ConceptDriftStream -s (generators.SEAGenerator -i SEED) -d (generators.SEAGenerator -i SEED -f 2) -p 25000 -w 1000) -d (ConceptDriftStream -s (generators.SEAGenerator -i SEED -f 3) -d (generators.SEAGenerator -i SEED -f 4) -p 25000 -w 1000) -p 25000 -w 1000)",
    "(ConceptDriftStream -s (ConceptDriftStream -s (generators.AgrawalGenerator -i SEED) -d (generators.AgrawalGenerator -i SEED -f 2) -p 25000 -w 0) -d (ConceptDriftStream -s (generators.AgrawalGenerator -i SEED -f 3) -d (generators.AgrawalGenerator -i SEED -f 4) -p 25000 -w 0) -p 25000 -w 0)",
    "(ConceptDriftStream -s (ConceptDriftStream -s (generators.AgrawalGenerator -i SEED) -d (generators.AgrawalGenerator -i SEED -f 2) -p 25000 -w 1000) -d (ConceptDriftStream -s (generators.AgrawalGenerator -i SEED -f 3) -d (generators.AgrawalGenerator -i SEED -f 4) -p 25000 -w 1000) -p 25000 -w 1000)",
    "(ConceptDriftStream -s (ConceptDriftStream -s (generators.AssetNegotiationGenerator -i SEED) -d (generators.AssetNegotiationGenerator -i SEED -f 2) -p 25000 -w 0) -d (ConceptDriftStream -s (generators.AssetNegotiationGenerator -i SEED -f 3) -d (generators.AssetNegotiationGenerator -i SEED -f 4) -p 25000 -w 0) -p 25000 -w 0)",
    "(ConceptDriftStream -s (ConceptDriftStream -s (generators.AssetNegotiationGenerator -i SEED) -d (generators.AssetNegotiationGenerator -i SEED -f 2) -p 25000 -w 1000) -d (ConceptDriftStream -s (generators.AssetNegotiationGenerator -i SEED -f 3) -d (generators.AssetNegotiationGenerator -i SEED -f 4) -p 25000 -w 1000) -p 25000 -w 1000)",
]


streams += [f"(ArffFileStream -f bases/{i})" for i in os.listdir("bases")]


for i in range(8):
    if not os.path.exists(f"results/{i}"):
        os.makedirs(f"results/{i}")
    for clf in clfs:
        for stream in streams:
            time.sleep(2)
            command = base_string_baseline.replace("CLF", clf)
            command = command.replace("STREAM", replace_seed(stream))
            if check_artificial(stream):
                command = command.replace("N_INSTANCES", "100000")
            else:
                command = command.replace("N_INSTANCES", "600000")
            red_stream = reduce_name(stream)
            output_file = f"results/{i}/clf={clf}&stream={red_stream}.csv"
            command = command.replace("OUTPUT_FILE", output_file.replace(" ", "_").replace("(", "").replace(")", ""))
            print(command)
            os.system(command)


    for stream in streams:
        for dcs in dcss:
            for bag in bags:
                for all_ in alls:
                    for init in inits:
                        for size in sizes:
                            time.sleep(2)
                            if not all_ and (bag or init):
                                continue
                            if all_:
                                command = base_string_proposed.replace("USE_ALL", " -a")
                            else:
                                command = base_string_proposed.replace("USE_ALL", " ")

                            if bag:
                                command = command.replace("USE_BAGGING", " -b")
                            else:
                                command = command.replace("USE_BAGGING", " ")

                            if init:
                                command = command.replace("USE_INIT", " -i")
                            else:
                                command = command.replace("USE_INIT", " ")

                            if check_artificial(stream):
                                command = command.replace("N_INSTANCES", "100000")
                            else:
                                command = command.replace("N_INSTANCES", "600000")
                            

                            command = command.replace("DCS_METHOD", dcs)
                            command = command.replace("STREAM", replace_seed(stream))
                            command = command.replace("SIZE", str(size))
                            red_stream = reduce_name(stream)
                            output_file = f"results/{i}/clf=manualBag&stream={red_stream}&all={all_}&bag={bag}&dcs={dcs}&init={init}&size={size}.csv"
                            command = command.replace("OUTPUT_FILE", output_file.replace(" ", "_").replace("(", "").replace(")", ""))
                            os.system(command)
