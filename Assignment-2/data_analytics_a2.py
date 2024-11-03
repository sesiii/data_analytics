# -*- coding: utf-8 -*-
"""Data_Analytics_A2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RsQJGg9JiyNChJxWGpIgPrhOKHboPtl4
"""

# prompt: load the dataset into a df
from itertools import combinations
from collections import defaultdict

import pandas as pd

# Assuming your dataset is in a CSV file named 'your_dataset.csv'
df = pd.read_csv('groceries.csv')

# Print the first 5 rows of the dataframe to verify it loaded correctly
df.head()

df.isnull().sum()

# Preprocess the dataset to create a list of transactions (sets of items)
transactions = [set(items.split(',')) for items in df['Items'].dropna()]

# Check the transactions list to confirm preprocessing
print("Sample transactions:", transactions[:5])  # Display the first few transactions for verification

# prompt: find the unique values among all the entries in df and tell their number

from collections import defaultdict

unique_items = set()
for transaction in transactions:
  for item in transaction:
    unique_items.add(item)

print("Unique items:", unique_items)
print("Number of unique items:", len(unique_items))

# Create a dictionary to store the one-hot encoded data
one_hot_data = defaultdict(list)

# Iterate through each transaction
for transaction in transactions:
  # Iterate through each unique item
  for item in unique_items:
    # If the item is present in the current transaction, add 1, otherwise add 0
    if item in transaction:
      one_hot_data[item].append(1)
    else:
      one_hot_data[item].append(0)

# Create a new DataFrame from the one-hot encoded data
one_hot_df = pd.DataFrame(one_hot_data)

# Print the one-hot encoded DataFrame
one_hot_df

# Calculate the number of 1s in each row
row_sums = one_hot_df.sum(axis=1)

# Print the results
print("Number of 1s in each row:\n", row_sums)

def func(data):
  for i in unique_items:
    if data[i] > 0:
      data[i] = i
  return data

data2 = one_hot_df.apply(func,axis=1)

data2.head()

newdata = data2.values

newdata

newdata.ndim

newdata = [i[i!=0].tolist() for i in newdata if i[i!=0].tolist()]

newdata[:10]

def generate_itemsets(transactions, min_support):
    """Generate frequent itemsets using the Apriori algorithm."""
    item_counts = defaultdict(int)
    transaction_count = len(transactions)

    # Count single items
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1

    # Filter single items by min_support
    current_itemsets = {itemset: count for itemset, count in item_counts.items()
                        if count / transaction_count >= min_support}

    # Store frequent itemsets
    frequent_itemsets = {}
    frequent_itemsets.update(current_itemsets)

    # Generate itemsets of size k > 1
    k = 2
    while current_itemsets:
        candidate_itemsets = defaultdict(int)
        itemsets_list = list(current_itemsets.keys())

        # Generate candidate itemsets of size k by joining pairs of size k-1 itemsets
        for i in range(len(itemsets_list)):
            for j in range(i + 1, len(itemsets_list)):
                candidate = itemsets_list[i] | itemsets_list[j]
                if len(candidate) == k:
                    # Count support for candidate itemset
                    candidate_count = sum(1 for transaction in transactions if candidate.issubset(transaction))
                    if candidate_count / transaction_count >= min_support:
                        candidate_itemsets[candidate] += candidate_count

        # Update frequent itemsets
        frequent_itemsets.update({itemset: count for itemset, count in candidate_itemsets.items() if count > 0})
        current_itemsets = candidate_itemsets
        k += 1

    return frequent_itemsets

from itertools import combinations

def generate_association_rules(frequent_itemsets, min_confidence):
    """Generate association rules from frequent itemsets."""
    rules = []

    for itemset, support_count in frequent_itemsets.items():
        # Generate all non-empty subsets of the itemset
        subsets = [frozenset(x) for i in range(1, len(itemset)) for x in combinations(itemset, i)]
        for subset in subsets:
            # Calculate the consequent
            consequent = itemset - subset

            if len(consequent) > 0:
                # Calculate support, confidence, and lift
                support = support_count
                support_consequent = sum(1 for k, v in frequent_itemsets.items() if consequent.issubset(k) and v > 0)
                
                # Debug: Print the current itemset, subset, and consequent
                print(f"Itemset: {itemset}, Subset: {subset}, Consequent: {consequent}")
                
                confidence = support / support_consequent if support_consequent > 0 else 0
                lift = confidence / (support_consequent / len(frequent_itemsets))

                if confidence >= min_confidence:
                    rules.append((set(subset), set(consequent), support, confidence, lift))

    return rules

# Input transactions
transactions = newdata
transactions

# Input minimum support and generate frequent itemsets
min_support = float(input("Enter the minimum support (e.g., 0.01): "))
frequent_itemsets = generate_itemsets(transactions, min_support)

# Print frequent itemsets and their support values
print("Frequent Itemsets and Support Values:")
for itemset, support_count in frequent_itemsets.items():
    support_value = support_count / len(transactions)  # Calculate support value
    print(f"{set(itemset)}: Support Value: {support_value:.4f}")

# Input minimum confidence and generate association rules
min_confidence = float(input("Enter the minimum confidence (e.g., 0.5): "))
rules = generate_association_rules(frequent_itemsets, min_confidence)

# Print association rules with their metrics
print("Association Rules:")
for rule in rules:
    antecedent, consequent, support, confidence, lift = rule
    print(f"Rule: {antecedent} -> {consequent}, Support: {support/len(transactions):.4f}, Confidence: {confidence:.4f}, Lift: {lift:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
def plot_top_n_frequent_itemsets(frequent_itemsets, n=5):
    """Visualize the top-N frequent itemsets using a bar chart."""
    # Sort frequent itemsets by support count
    sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda x: x[1], reverse=True)

    # Get top N itemsets
    top_n_itemsets = sorted_itemsets[:n]

    # Prepare data for plotting
    itemset_labels = [str(set(itemset)) for itemset, _ in top_n_itemsets]
    support_values = [count for _, count in top_n_itemsets]

    # Create bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=support_values, y=itemset_labels, hue=itemset_labels, palette='viridis', dodge=False, legend=False)
    plt.title(f'Top-{n} Frequent Itemsets')
    plt.xlabel('Support Count')
    plt.ylabel('Itemsets')
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_top_n_strongest_rules(rules, n=5):
    """Visualize the top-N strongest rules using a scatter plot with jitter and print rule details."""
    # Sort rules by lift and select the top-N
    sorted_rules = sorted(rules, key=lambda x: x[4], reverse=True)
    top_n_rules = sorted_rules[:n]

    # Prepare data for plotting
    antecedents = [' & '.join(map(str, rule[0])) for rule in top_n_rules]
    consequents = [' & '.join(map(str, rule[1])) for rule in top_n_rules]
    supports = [rule[2] / len(transactions) for rule in top_n_rules]  # Normalized support
    confidences = [rule[3] for rule in top_n_rules]
    lifts = np.array([rule[4] for rule in top_n_rules])

    # Add a small jitter to prevent overlapping points
    supports_jittered = np.array(supports) + np.random.normal(0, 0.000001, size=len(supports))
    lifts_jittered = lifts + np.random.normal(0, 0.000001, size=len(lifts))

    # Create scatter plot with uniform mark points
    plt.figure(figsize=(20, 10))
    sns.scatterplot(x=lifts_jittered, y=supports_jittered, s=100, marker='o', color='b', alpha=0.6)

    # Annotate points with rule details
    for i in range(len(top_n_rules)):
        plt.annotate(f"{antecedents[i]} -> {consequents[i]}", (lifts_jittered[i], supports_jittered[i]), fontsize=9, ha='right')

    plt.title(f'Top-{n} Strongest Rules (by Lift)')
    plt.xlabel('Lift')
    plt.ylabel('Normalized Support')
    plt.grid()
    plt.show()

    # Print rule details separately
    print(f"Top {n} Strongest Rules:")
    print(f"{'Antecedent':<50} {'Consequent':<50} {'Support':<10} {'Confidence':<10} {'Lift':<10}")
    print("="*120)
    for i in range(n):
        print(f"{antecedents[i]:<50} {consequents[i]:<50} {supports[i]:<10.6f} {confidences[i]:<10.4f} {lifts[i]:<10.4f}")

def summarize_results(frequent_itemsets, rules):
    """Generate a detailed report summarizing the frequent patterns and association rules."""
    print("\nFrequent Itemsets Summary:")
    for itemset, count in frequent_itemsets.items():
        support_value = count / len(transactions)
        print(f"Itemset: {set(itemset)}, Support Count: {count}, Support Value: {support_value:.4f}")

    print("\nAssociation Rules Summary:")
    for rule in rules:
        antecedent, consequent, support, confidence, lift = rule
        print(f"Rule: {antecedent} -> {consequent}, Support: {support/len(transactions):.6f}, Confidence: {confidence:.4f}, Lift: {lift:.4f}")

# Plot Top 10 Frequent Itemsets
plot_top_n_frequent_itemsets(frequent_itemsets, n=10)

# Plot Top 10 Strongest Rules
plot_top_n_strongest_rules(rules, n=5)

"""# Bonus Part"""

import itertools

def generate_association_rules(frequent_itemsets, transactions, min_confidence, min_leverage=None, min_conviction=None):
    """Generate association rules with support, confidence, lift, leverage, and conviction."""
    rules = []
    transaction_count = len(transactions)

    # Calculate support for individual items
    item_supports = {itemset: support / transaction_count for itemset, support in frequent_itemsets.items()}

    # Generate rules for each frequent itemset
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue  # Skip if itemset cannot form a rule

        for antecedent in itertools.chain.from_iterable(itertools.combinations(itemset, r) for r in range(1, len(itemset))):
            antecedent = frozenset(antecedent)
            consequent = itemset - antecedent

            if consequent:
                antecedent_support = item_supports[antecedent]
                consequent_support = item_supports[consequent]
                rule_support = frequent_itemsets[itemset] / transaction_count
                confidence = rule_support / antecedent_support
                lift = confidence / consequent_support

                # Calculate leverage and conviction
                leverage = rule_support - (antecedent_support * consequent_support)
                # Avoid infinity for conviction by setting a high value if confidence is 1
                conviction = ((1 - consequent_support) / (1 - confidence)) if confidence < 1 else 10**6

                # Apply minimum confidence and additional thresholds if specified
                if confidence >= min_confidence:
                    if (min_leverage is None or leverage >= min_leverage) and \
                       (min_conviction is None or conviction >= min_conviction):
                        rules.append((antecedent, consequent, rule_support, confidence, lift, leverage, conviction))

    return rules

# Example usage
# Assume 'frequent_itemsets' is a dictionary containing itemsets as keys and their support counts as values
# Assume 'transactions' is a list of transactions (each transaction is a set of items)

# User inputs
min_confidence = float(input("Enter minimum confidence (e.g., 0.1): "))
min_leverage = float(input("Enter minimum leverage (e.g., 0.005) or leave blank: ") or 0.01)
min_conviction = float(input("Enter minimum conviction (e.g., 0.8) or leave blank: ") or 1.2)

# Generate association rules with specified thresholds
rules = generate_association_rules(frequent_itemsets, transactions, min_confidence, min_leverage, min_conviction)

# Print the generated rules
print("\nGenerated Association Rules with Leverage and Conviction:")
for rule in rules:
    antecedent, consequent, support, confidence, lift, leverage, conviction = rule
    print(f"Rule: {set(antecedent)} -> {set(consequent)}")
    print(f"  Support: {support:.4f}, Confidence: {confidence:.4f}, Lift: {lift:.4f}, Leverage: {leverage:.4f}, Conviction: {conviction:.4f}")
    print("="*60)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_top_n_strongest_rules_with_jitter(rules, transactions, n=10):
    """Visualize the top-N strongest rules using a scatter plot with jitter to avoid overlapping points and print details."""
    # Sort rules by lift and select the top-N
    sorted_rules = sorted(rules, key=lambda x: x[4], reverse=True)[:n]  # x[4] is lift
    antecedents = [' & '.join(map(str, rule[0])) for rule in sorted_rules]
    consequents = [' & '.join(map(str, rule[1])) for rule in sorted_rules]
    lifts = [rule[4] for rule in sorted_rules]
    leverages = [rule[5] for rule in sorted_rules]
    convictions = [rule[6] for rule in sorted_rules]

    # Add a small jitter to lift and leverage to avoid overlapping
    lifts_jittered = np.array(lifts) + np.random.normal(0, 0.0005, size=len(lifts))
    leverages_jittered = np.array(leverages) + np.random.normal(0, 0.0005, size=len(leverages))

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x=lifts_jittered, y=leverages_jittered, s=200, alpha=0.7, marker='o', color='blue')  # Fixed marker symbol
    plt.xlabel("Lift")
    plt.ylabel("Leverage")
    plt.title(f"Top-{n} Strongest Rules by Lift (with Leverage and Conviction)")

    # Annotate each point with rule info
    for i in range(len(sorted_rules)):
        plt.annotate(f"{antecedents[i]} -> {consequents[i]}", (lifts_jittered[i], leverages_jittered[i]), fontsize=9, ha='right')

    plt.grid(True)
    plt.show()

    # Print rule details separately
    print(f"Top {n} Strongest Rules:")
    print(f"{'Antecedent':<50} {'Consequent':<50} {'Lift':<10} {'Leverage':<10} {'Conviction':<10}")
    print("="*120)
    for i in range(len(sorted_rules)):
        print(f"{antecedents[i]:<50} {consequents[i]:<50} {lifts[i]:<10.4f} {leverages[i]:<10.4f} {convictions[i]:<10.4f}")

# Example usage
plot_top_n_strongest_rules_with_jitter(rules, transactions, n=10)



"""# Part - II"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

"""# Defining the FP-Tree Structure
This snippet defines the FP-tree data structure used for efficient itemset mining. The FPTree class builds the tree from the transactions, and TreeNode represents each node in the tree.
"""

class HashTreeNode:
    def __init__(self, max_leaf_size=3, max_depth=5, depth=0):
        self.is_leaf = True
        self.itemsets = []
        self.children = {}
        self.max_leaf_size = max_leaf_size
        self.max_depth = max_depth
        self.depth = depth

    def insert(self, itemset):
        sorted_itemset = sorted(itemset)
        if self.is_leaf:
            self.itemsets.append(sorted_itemset)
            if len(self.itemsets) > self.max_leaf_size and self.depth < self.max_depth:
                self._split()
        else:
            if self.depth < len(sorted_itemset):  # Check if depth is valid
                hash_key = self._hash(sorted_itemset[self.depth])
                if hash_key not in self.children:
                    self.children[hash_key] = HashTreeNode(self.max_leaf_size, self.max_depth, self.depth + 1)
                self.children[hash_key].insert(sorted_itemset)

    def _split(self):
        self.is_leaf = False
        for itemset in self.itemsets:
            if self.depth < len(itemset):  # Check if depth is valid
                hash_key = self._hash(itemset[self.depth])
                if hash_key not in self.children:
                    self.children[hash_key] = HashTreeNode(self.max_leaf_size, self.max_depth, self.depth + 1)
                self.children[hash_key].insert(itemset)
        self.itemsets = []

    def _hash(self, item):
        return hash(item) % 3  # Simple hash function

    def search(self, transaction, k):
        if self.is_leaf:
            return [set(itemset) for itemset in self.itemsets if set(itemset).issubset(transaction)]
        else:
            if k < len(transaction):
                hash_key = self._hash(transaction[k])
                if hash_key in self.children:
                    return self.children[hash_key].search(transaction, k + 1)
        return []

"""# Generating Frequent Itemsets
This snippet generates frequent itemsets from the transactions using the FP-tree.
"""

def apriori(transactions, min_support):
    root = HashTreeNode()
    item_count = {}

    # Count occurrences of individual items
    for transaction in transactions:
        for item in transaction:
            item_count[item] = item_count.get(item, 0) + 1

    # Filter items by min support
    frequent_items = {item for item, count in item_count.items() if count >= min_support}

    # Insert frequent items into Hash Tree
    for transaction in transactions:
        transaction_items = set(transaction).intersection(frequent_items)
        root.insert(transaction_items)

    frequent_itemsets = []
    support_data = {}

    # Retrieve frequent itemsets and their support counts
    def retrieve_frequent_itemsets(node):
        if node.is_leaf:
            for itemset in node.itemsets:
                support_count = sum(1 for transaction in transactions if set(itemset).issubset(transaction))
                frequent_itemsets.append(itemset)
                support_data[tuple(itemset)] = support_count
        else:
            for child in node.children.values():
                retrieve_frequent_itemsets(child)

    retrieve_frequent_itemsets(root)

    return frequent_itemsets, support_data

"""# Generating Association Rules
This snippet generates association rules from the frequent itemsets, including support, confidence, and profitability calculations.
"""

def generate_association_rules(frequent_itemsets, support_data, transactions, min_confidence):
    rules = []

    for itemset in frequent_itemsets:
        support_count = support_data[tuple(itemset)]
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = set(antecedent)
                consequent = set(itemset) - antecedent
                if len(consequent) > 0:
                    antecedent_support = support_data.get(tuple(antecedent), 0)
                    if antecedent_support > 0:
                        confidence = support_count / antecedent_support
                        if confidence >= min_confidence:
                            lift = confidence / (support_data[tuple(consequent)] / len(transactions))
                            rules.append((antecedent, consequent, support_count, confidence, lift))
    return rules

"""# Pruning Rules Based on Profitability
This snippet prunes the generated rules based on a minimum profitability threshold.
"""

import matplotlib.pyplot as plt
import pandas as pd

def visualize_frequent_itemsets(frequent_itemsets, support_data, top_n=10):
    sorted_itemsets = sorted(frequent_itemsets, key=lambda x: support_data[tuple(x)], reverse=True)[:top_n]
    supports = [support_data[tuple(itemset)] for itemset in sorted_itemsets]

    plt.barh(range(len(sorted_itemsets)), supports, align='center')
    plt.yticks(range(len(sorted_itemsets)), [str(set(itemset)) for itemset in sorted_itemsets])
    plt.xlabel('Support')
    plt.title('Top-N Frequent Itemsets')
    plt.show()

def visualize_strongest_rules(rules, top_n=10):
    sorted_rules = sorted(rules, key=lambda x: x[4], reverse=True)[:top_n]  # Sort by lift
    antecedents = [' & '.join(map(str, rule[0])) for rule in sorted_rules]
    consequents = [' & '.join(map(str, rule[1])) for rule in sorted_rules]
    lifts = [rule[4] for rule in sorted_rules]

    df = pd.DataFrame({'Antecedent': antecedents, 'Consequent': consequents, 'Lift': lifts})
    df.plot.scatter(x='Antecedent', y='Lift', alpha=0.5)
    plt.title('Top-N Strongest Rules by Lift')
    plt.show()

if __name__ == "__main__":

    # User inputs
    min_support = float(input("Enter minimum support count (e.g., 0.02): "))
    min_confidence = float(input("Enter minimum confidence (e.g., 0.6): "))
    top_n = int(input("Enter the number of top frequent itemsets and rules to visualize: "))

    # Run Apriori algorithm
    frequent_itemsets, support_data = apriori(transactions, min_support)

    # Generate association rules
    rules = generate_association_rules(frequent_itemsets, support_data, transactions, min_confidence)

    # Visualize results
    visualize_frequent_itemsets(frequent_itemsets, support_data, top_n)
    visualize_strongest_rules(rules, top_n)

    # Print results
    print("Frequent Itemsets:")
    for itemset in frequent_itemsets:
        print(f"{set(itemset)}: {support_data[tuple(itemset)]}")

    print("\nAssociation Rules:")
    for rule in rules:
        print(f"Rule: {set(rule[0])} -> {set(rule[1])}, Support: {rule[2]}, Confidence: {rule[3]}, Lift: {rule[4]}")

"""# Plotting Top-N Strongest Rules
This snippet visualizes the top-N strongest rules using a scatter plot and prints the details of each rule.
"""

# Parameters
min_support = 0.009
min_confidence = 0.001
min_lift = 1.2

# Run Apriori algorithm
frequent_itemsets, support_data = apriori(transactions, min_support)

# Generate association rules
rules = generate_association_rules(frequent_itemsets, support_data, transactions, min_confidence)

# Filter rules based on the minimum confidence and lift thresholds
def filter_rules_by_metrics(rules, min_confidence=0.0, min_lift=0.0):
    """Filter rules based on minimum confidence and lift thresholds."""
    filtered_rules = []
    for rule in rules:
        antecedent, consequent, support, confidence, lift = rule
        if confidence >= min_confidence and lift >= min_lift:
            filtered_rules.append(rule)
    return filtered_rules

filtered_rules = filter_rules_by_metrics(rules, min_confidence=min_confidence, min_lift=min_lift)


# Plot visualizations
plot_top_n_frequent_itemsets(support_data, n=5)
plot_top_n_strongest_rules_with_jitter(filtered_rules, transactions, n=5)

# User inputs
min_support = float(input("Enter minimum support threshold (e.g., 0.01): "))  # Minimum support threshold
min_confidence = float(input("Enter minimum confidence (e.g., 0.6): "))  # Minimum confidence

# Generate frequent itemsets based on minimum support
frequent_itemsets, support_data = apriori(transactions, min_support)

# Generate association rules from the frequent itemsets with minimum confidence
rules = generate_association_rules(frequent_itemsets, support_data, transactions, min_confidence)
# Print the generated rules with their metrics
print("Generated Rules:")
for rule in rules:
    antecedent, consequent, support, confidence, lift = rule
    print(f"Rule: {antecedent} -> {consequent}, Support: {support:.4f}, Confidence: {confidence:.4f}, Lift: {lift:.4f}")

transactions

