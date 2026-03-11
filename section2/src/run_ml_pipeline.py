"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import argparse

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"/teamspace/studios/this_studio/nd320-c3-3d-imaging-starter/section1/out"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "/teamspace/studios/this_studio/nd320-c3-3d-imaging-starter/section2/out"
        # STAND-OUT: single_class mode merges anterior+posterior into one class
        self.single_class = False
        self.num_classes = 3  # 0=bg, 1=anterior, 2=posterior (default)

if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser(description="Train hippocampus segmentation model")
    parser.add_argument("--single-class", action="store_true",
                        help="STAND-OUT: Merge anterior+posterior labels into single hippocampus class")
    args = parser.parse_args()

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()

    if args.single_class:
        c.single_class = True
        c.num_classes = 2  # 0=bg, 1=hippocampus (merged)
        c.name = "Basic_unet_single_class"

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)

    # STAND-OUT: If single_class mode, merge anterior (1) and posterior (2) into one class (1)
    if c.single_class:
        print("Single-class mode: merging anterior and posterior labels into one class")
        for sample in data:
            sample["seg"] = (sample["seg"] > 0).astype(int)


    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    # Use roughly 70/10/20 split
    total = len(keys)
    train_end = int(0.7 * total)
    val_end = int(0.8 * total)

    split["train"] = list(range(0, train_end))
    split["val"] = list(range(train_end, val_end))
    split["test"] = list(range(val_end, total))

    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # run training
    exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

