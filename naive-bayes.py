# Define the training data
training_data = {
    'mango': {'yellow': 350, 'sweet': 450, 'long': 0, 'total': 650},
    'banana': {'yellow': 400, 'sweet': 300, 'long': 350, 'total': 400},
    'others': {'yellow': 50, 'sweet': 100, 'long': 50, 'total': 150}
}
# Calculate class priors
total_fruits = sum([training_data[label]['total'] for label in training_data])
class_priors = {label: training_data[label]['total'] / total_fruits for label in training_data}

# Define a function to calculate the conditional probabilities
def calculate_conditional_probabilities(feature, value, label):
    count_label = training_data[label]['total']
    count_feature = training_data[label][feature]
    # Laplace smoothing to avoid zero probabilities
    conditional_prob = (count_feature) / (count_label)
    return conditional_prob

# Define a function to predict the class of a fruit
def predict_fruit(properties):
    probabilities = {}
    for label in training_data:
        probabilities[label] = class_priors[label]
        for feature, value in properties.items():
            probabilities[label] *= calculate_conditional_probabilities(feature, value, label)
    predicted_label = max(probabilities, key=probabilities.get)
    return predicted_label, probabilities

# Test the classifier with example properties
fruit_properties = {'yellow': 500, 'sweet': 600, 'long': 200}
predicted_fruit, probabilities = predict_fruit(fruit_properties)
print("Predicted fruit:", predicted_fruit)

# Display calculated probabilities
print("Calculated Probabilities:")
for label, probability in probabilities.items():
    print(f"{label}: {probability}")