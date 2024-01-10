import sys
import mne
import numpy as np
from enums.process import Process
from utils.Experiment import Experiment
 

def train_all_models():
    accuracies = []
    for run in range(1, 7):
        experiment_means = []
        for subject in range(1, 110):
            try:
                experiment = Experiment(str(subject), str(run), True)
                _, accuracy = experiment.train()
                print(f"experiment {run}: subject {subject} = {accuracy}")
                experiment_means.append(accuracy)
            except Exception as e:
                print(f"Error occures during training on subject {subject}, experiment {run}: {e}")
        experiment_mean = np.mean(experiment_means)
        accuracies.append(experiment_mean)
        print(f"experiment {run}:    accuracy = {experiment_mean}")
        print('-----------------------------------')
    
    print("Mean accuracy of the six different experiments for all 100 subjects:")
    for i in range(len(accuracies)):
        print(f"experiment {i + 1}:    accuracy = {accuracies[i]}")
    print(f"Mean accuracy of 6 experiments: {np.mean(accuracies)}")


def train_one_model(experiment):
    scores, _ = experiment.train()
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
    try:
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
    except Exception as e:
        print(e)
        return
    

if __name__ == "__main__":
    main()