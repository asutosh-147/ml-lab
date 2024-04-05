import pandas as pd
def sequential_covering(examples):
    # Create a list of rules
    rules = []
    remaining_examples = examples.copy()
    while not remaining_examples.empty:
        # Create a rule for the current example
        rule = generate_rule(remaining_examples)
        # Add the rule to the list
        if rule not in rules:
            rules.append(rule)
        else:
            break
        # Remove the examples covered by the rule
        remaining_examples = remove_covered_examples(remaining_examples, rule)
    return rules

def generate_rule(examples):
    rule = {}
    for attribute in examples.columns:
        # Add the attribute to the rule
        value = examples[attribute].mode()[0]
        rule[attribute] = value
    return rule

def remove_covered_examples(examples, rule):
    for i, row in enumerate(examples):
        flag = True
        for attribute, value in rule.items():
            flag = flag and examples[attribute].loc[examples.index[0]] == value
            if flag:
                examples = examples.drop(examples.index[i])
    return examples

data = {
    'outlook' : ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'temperature' : ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'humidity' : ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'windy' : ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'play' : ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)
rules = sequential_covering(df)
for i, rule in enumerate(rules):
    print(f"Rule {i + 1}: {rule}")