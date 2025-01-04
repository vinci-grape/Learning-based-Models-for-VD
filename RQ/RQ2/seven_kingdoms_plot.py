import matplotlib.pyplot as plt

models = ['devign', 'reveal', 'ivdetect', 'LineVul', 'SVulD', 'chatgpt-icl-same-repo']
categories = ['Category A', 'Category B', 'Category C']

accuracy_data = {
    'devign': [0.8, 0.9, 0.75],
    'reveal': [0.7, 0.85, 0.65],
    'ivdetect': [0.6, 0.75, 0.7],
    'LineVul': [0.9, 0.85, 0.8],
    'SVulD': [0.75, 0.8, 0.7],
    'chatgpt-icl-same-repo': [0.85, 0.9, 0.8],
}

sorted_models = sorted(models, key=lambda x: accuracy_data[x][0], reverse=True)

fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.tab20c.colors

for i, model in enumerate(sorted_models):
    ax.plot(categories, accuracy_data[model], marker='o', color=colors[i], label=model)

ax.set_title('Accuracy Performance on Different Categories')
ax.set_xlabel('Categories')
ax.set_ylabel('Accuracy')
ax.legend()

plt.tight_layout()

plt.show()
