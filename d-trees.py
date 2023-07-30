import pandas as pd
import numpy as np


dataset_name = "agaricus-lepiota.data"
column_names = [
    "toxicity", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
    "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]
total_data = pd.read_csv(dataset_name, header=None, names=column_names)

depth = 0

def calc_entropy(dataset):
    labels = dataset["toxicity"].value_counts()
    probabilities = labels / labels.sum()
    entropy_value = -sum(probabilities * np.log2(probabilities))
    return entropy_value


def calc_conditional_entropy(dataset, feature):
    attributes = dataset[feature].unique()
    total_entropy = 0

    for attribute in attributes:
        subset = filter_on_attribute(dataset, feature, attribute)
        weight = len(subset) / len(dataset)
        subset_entropy = calc_entropy(subset)
        total_entropy += weight * subset_entropy

    return total_entropy


def filter_on_attribute(dataset, col_name, attribute):
    return dataset[dataset[col_name] == attribute]


def information_gain(dataset, feature):
    target_entropy = calc_entropy(dataset)
    feature_entropy = calc_conditional_entropy(dataset, feature)
    information_gain_value = target_entropy - feature_entropy
    return information_gain_value


def find_highest_info_gain(dataset):
    info_gains = {}
    for col_num, col_name in enumerate(dataset.columns[1:]):
        info_gains[col_num] = information_gain(dataset, col_name)

    info_gains = [col_num + 1
                  for (col_num, info_gain) in sorted(info_gains.items(), key=lambda x: x[1], reverse=True)]

    best_attribute_col = info_gains[0]
    best_attribute = dataset.columns[best_attribute_col]
    return best_attribute_col, best_attribute


def train(dataset):
    global depth

    if len(dataset["toxicity"].unique()) == 1:
        return f"Entropy: 0   Only {dataset['toxicity'].iloc[0]}"
    elif len(dataset.columns) == 1:
        return dataset["toxicity"].value_counts().idxmax()
    else:
        best_attribute_col, best_attribute = find_highest_info_gain(dataset)
        print(f"Best Split: {best_attribute}")
        depth += 1

        decision_tree = {best_attribute: {}}

        attributes = dataset[best_attribute].unique()
        for attribute in attributes:
            subset = dataset[dataset[best_attribute] == attribute]
            subset = subset.drop(columns=[best_attribute])

            decision_tree[best_attribute][attribute] = train(subset)



print(train(total_data))
print(depth)