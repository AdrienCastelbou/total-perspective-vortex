from enum import Enum, EnumMeta

class MyProcessMeta(EnumMeta):  
    def __contains__(cls, item): 
        return item in cls.__members__.values()

class Process(str, Enum, metaclass=MyProcessMeta):
    PREPROCESS  = 'preprocess'
    TRAIN       = 'train'
    TEST        = 'test'