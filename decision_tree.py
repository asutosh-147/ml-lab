import pandas as pd
import numpy as np

def entropy(data):
    labels = data.iloc[:,-1]
    counts = labels.value_counts(normalize = True)
    return -np.sum(counts * np.log2(counts))

def information_gain(data, attribute):
    values = data[attribute].unique()
    gain = entropy(data)
    for value in values:
        subset = data[data[attribute] == value]
        gain -= len(subset) / len(data) * entropy(subset)
    return gain

def id3(data, target, attributes):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]
    if len(attributes) == 0:
        return data[target].mode()[0]
    gains = {attribute: information_gain(data, attribute) for attribute in attributes}
    best_attribute = max(gains, key = gains.get)
    tree = {best_attribute: {}}
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        tree[best_attribute][value] = id3(subset, target, [a for a in attributes if a != best_attribute])
    return tree

data = {
    'outlook' : ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'temperature' : ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'humidity' : ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'windy' : ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'play' : ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}
df = pd.DataFrame(data)
attributes = df.columns[:-1]
tree = id3(df, 'play', attributes)
print(tree)
