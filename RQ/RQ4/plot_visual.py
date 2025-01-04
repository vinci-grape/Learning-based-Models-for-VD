import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = {
  'LineVul': {'Function Call': 3764, 'Field Expression': 3559, 'Assignment Operation': 1439, 'Relational Operation': 1084, 'If Statement': 967, 'Declaration Statement': 756, 'Arithmetic Operation': 717, 'Logical Operation': 512, 'Bitwise Operation': 458, 'Return Statement': 222 , 'For Statement' : 89, 'Case Statement':35 , 'While Statement':26 , 'Jump Statement':18 , 'Switch Statement':8},
  'SVulD': {'Function Call': 3755, 'Field Expression': 3661, 'Assignment Operation': 1483, 'Relational Operation': 1106, 'If Statement': 951, 'Declaration Statement': 942, 'Arithmetic Operation': 783, 'Logical Operation': 441, 'Bitwise Operation': 401, 'Return Statement': 202 , 'For Statement' : 107, 'Case Statement':34 , 'While Statement':29 , 'Jump Statement':20 , 'Switch Statement':5},
  'Devign': {'Field Expression': 4346, 'Function Call': 3718, 'Relational Operation': 1545, 'If Statement': 1514, 'Assignment Operation': 1453, 'Declaration Statement': 1138, 'Arithmetic Operation': 924, 'Logical Operation': 671, 'Bitwise Operation': 540, 'Return Statement': 226, 'For Statement' : 146, 'While Statement':54 ,'Switch Statement':28,'Case Statement':21 ,  'Jump Statement':11 , },
  'Reveal': {'Field Expression': 4314, 'Function Call': 3725, 'Assignment Operation': 1963, 'Relational Operation': 1306, 'If Statement': 1265, 'Arithmetic Operation': 1007, 'Logical Operation': 581, 'Declaration Statement': 493, 'Bitwise Operation': 479, 'Return Statement': 366 ,'For Statement' : 138, 'Case Statement':42 , 'While Statement':33 , 'Jump Statement':19 , 'Switch Statement':8},
  'IVdetect': {'Function Call': 3299, 'Field Expression': 3279, 'If Statement': 1685, 'Relational Operation': 1489, 'Assignment Operation': 1310, 'Arithmetic Operation': 698, 'Logical Operation': 564, 'Declaration Statement': 419, 'Bitwise Operation': 390, 'Return Statement': 372, 'For Statement' : 135, 'Case Statement':71 , 'While Statement':65 , 'Jump Statement':20 , 'Switch Statement':13}
}

operation_types = list(data['LineVul'].keys())

colors = sns.color_palette("Set2", len(data)+1)
# colors.pop(4)

bar_width = 0.16
x = np.arange(len(operation_types))
plt.figure(figsize=(9, 6))
for i, (model, module_data) in enumerate(data.items()):
    values = [module_data[op_type] for op_type in operation_types]
    hatch = ''
    if model in ['LineVul' , 'SVulD'] :
        hatch= '\\\\\\'#
    plt.bar(x + i * bar_width, values, bar_width, label=model, color=colors[i], hatch=hatch , edgecolor='#2b2d30' , linewidth = 0.5)

plt.xticks(x + bar_width * (len(data) - 1 )* 0.5  , operation_types, rotation=45)
plt.legend()
plt.savefig('statement_types.svg',bbox_inches="tight")
plt.show()