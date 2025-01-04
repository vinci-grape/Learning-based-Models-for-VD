import subprocess
from dataclasses import dataclass


@dataclass
class Setting:
    p_which_test_set: str
    p_temperature: float
    p_mode: str
    p_icl_strategy: str
    p_cot_strategy: str


settings = [
    Setting('raw', 0.0, 'zero_shot', 'Fix', 'FewShotByDiversity'),
    Setting('raw', 0.0, 'icl', 'Fix', 'FewShotByDiversity'),
    Setting('raw', 0.0, 'icl', 'Random', 'FewShotByDiversity'),
    Setting('raw', 0.0, 'icl', 'Diversity', 'FewShotByDiversity'),
    Setting('raw', 0.0, 'icl', 'CosineSimilar', 'FewShotByDiversity'),
    Setting('raw', 0.0, 'icl', 'SameRepo', 'FewShotByDiversity'),
    Setting('raw', 0.0, 'cot', 'Fix', 'ZeroShot'),
    Setting('raw', 0.0, 'cot', 'Fix', 'FewShotByHand'),
    Setting('raw', 0.0, 'cot', 'Fix', 'FewShotByDiversity'),
    Setting('top_cwe_25', 0.0, 'zero_shot', 'Fix', 'FewShotByDiversity'),
    Setting('top_cwe_25', 0.0, 'icl', 'Fix', 'FewShotByDiversity'),
    Setting('top_cwe_25', 0.0, 'icl', 'Random', 'FewShotByDiversity'),
    Setting('top_cwe_25', 0.0, 'icl', 'Diversity', 'FewShotByDiversity'),
    Setting('top_cwe_25', 0.0, 'icl', 'CosineSimilar', 'FewShotByDiversity'),
    Setting('top_cwe_25', 0.0, 'icl', 'SameRepo', 'FewShotByDiversity'),
]


if __name__ == '__main__':

    for s in settings:

        subprocess.call(f'python chatgpt.py -p_which_test_set={s.p_which_test_set} -p_temperature={s.p_temperature} -p_mode={s.p_mode} -p_icl_strategy={s.p_icl_strategy} -p_cot_strategy={s.p_cot_strategy}',
                         shell=True, )
