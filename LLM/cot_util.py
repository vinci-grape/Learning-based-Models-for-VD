from enum import Enum

class CotStrategy(Enum):
    ZeroShot = "zero_shot"
    FewShotByHand = "few_shot_by_hand"  
    FewShotByDiversity = "few_shot_by_diversity" 

