import sys
import mne
import numpy as np
from enums.process import Process
from utils.Experiment import Experiment
 

def train_all_models():
    means = []
    for run in range(3, 7):
        experiment_means = []
        for subject in range(1, 39):
            experiment = Experiment(str(subject), str(run), True)
            scores = experiment.train()
            mean = np.mean(scores)
            print(f"experiment {run}: subject {subject} = {mean}")
            experiment_means.append(mean)
        experiment_mean = np.mean(experiment_means)
        means.append(experiment_mean)
        print(f"experiment {run}:    accuracy = {experiment_mean}")
        print('-----------------------------------')
    
    print("Mean accuracy of the six different experiments for all 100 subjects:")
    for i in range(len(means)):
        print(f"experiment {i}:    accuracy = {means[i]}")
    print(f"Mean accuracy of 6 experiments: {np.mean(means)}")


def train_one_model(experiment):
    scores = experiment.train()
    print(scores)
    print(f"cross_val_score: {np.mean(scores)}")


def get_process():
    args = sys.argv[1:]

    if len(args) == 0:
        return Process.ALL, None
    elif len(args) != 3 or args[2] not in Process:
        raise Exception("Error: wrong usage -> python mybci.py [subject id (int)] [experiment id (int)] [process (preprocess/train/predict)]\nExample : python mybci.py 1 14 train")
    getAllRuns = args[2] == Process.TRAIN
    experiment = Experiment(args[0], args[1], getAllRuns)
    return args[2], experiment

def run_prediction(experiment):
    if (experiment.get_model() == False):
        raise Exception("model file does not exist")
    experiment.predict()


def main():
    #try:
        mne.set_log_level('WARNING')
        process, experiment = get_process()
        if process == Process.PREPROCESS:
           experiment.preprocess_data(True)
        elif process == Process.TRAIN:
            train_one_model(experiment)
        elif process == Process.ALL:
            train_all_models()
        elif process == Process.PREDICT:
            run_prediction(experiment)
    #except Exception as e:
    #    print(e)
        return
    

if __name__ == "__main__":
    main()