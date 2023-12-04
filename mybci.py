import sys
import os.path
from enums.process import Process
from process.Preprocess import Preprocess

def get_path(subject, experiment):
    subjectId = subject.zfill(3)
    experimentId = experiment.zfill(2)
    filename = f"./physionet.org/files/eegmmidb/1.0.0/S{subjectId}/S{subjectId}R{experimentId}.edf"
    if (not os.path.isfile(filename)):
        raise Exception()
    return filename


def get_params():
    args = sys.argv[1:]

    if len(args) == 0:
        return "all", None
    elif len(args) != 3 or args[2] not in Process:
        raise Exception()
    filename = get_path(args[0], args[1])
    return args[2],filename

def main():
    try:
        process, filename = get_params()
    except:
        print("Error: wrong usage -> python mybci.py [subject id (int)] [experiment id (int)] [process (preprocess/train/predict)]\nExample : python mybci.py 1 14 train")
    if process == Process.PREPROCESS:
        preprocessor = Preprocess(vizualize=True)
        preprocessor.run(filename)

if __name__ == "__main__":
    main()