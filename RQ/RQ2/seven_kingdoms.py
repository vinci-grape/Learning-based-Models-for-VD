import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = {
     'LineVul': {'Input Validation and Representation': 203, 'Code Quality': 91, 'Security Features': 21, 'API Abuse': 3, },
     'SVulD': {'Input Validation and Representation': 152, 'Code Quality': 69, 'Security Features': 19, 'API Abuse': 1,},
     'Devign': {'Input Validation and Representation': 202, 'Code Quality': 76, 'Security Features': 14, 'API Abuse': 2,},
     'Reveal': {'Input Validation and Representation': 207, 'Code Quality': 78, 'Security Features': 20, 'API Abuse': 2,},
     'IVdetect': {'Input Validation and Representation': 207, 'Code Quality': 89, 'Security Features': 18, 'API Abuse': 1, }
}

operation_types = list(data['LineVul'].keys())
operation_types_label = operation_types

colors = sns.color_palette(['#558ed5','#93cddd','#c3d69b','#fac090','#d99694'])
# colors.pop(4)

bar_width = 0.15
x = np.arange(len(operation_types)) * 1.0
plt.figure(figsize=(9, 6))
for i, (module, module_data) in enumerate(data.items()):
    values = [module_data[op_type] for op_type in operation_types]
    plt.bar(x + i * bar_width, values, bar_width, label=module, color=colors[i])

operation_types_label[0] = 'Input Validation\n and Representation'
plt.xticks(x + bar_width * 2, operation_types,rotation=0)
plt.legend()
plt.savefig("kingdons.jpg",bbox_inches="tight")

plt.show()
