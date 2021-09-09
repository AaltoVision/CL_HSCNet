from .seven_scenes import SevenScenes
from .twelve_scenes import TwelveScenes

def get_dataset(name):

    return {
            '7S' : SevenScenes,
            '12S' : TwelveScenes  
           }[name]
