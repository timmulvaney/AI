from globals import * 

# show some stuff about the penguins
def basic(df):
  df.columns
  df.head()
  df.describe()
  df.info()

# baseline here? 

# A baseline method for classification is a simple model or technique used as a point of reference for comparing the performance of more complex models. It serves as a benchmark against which the performance of more sophisticated models can be evaluated. Baseline methods are typically straightforward and easy to implement, providing a basic level of performance that any more advanced model should aim to surpass.

# Common baseline methods for classification include:

#     Majority Class Classifier: This method simply predicts the most frequent class in the training data for every instance in the test data. It's useful for imbalanced datasets where one class dominates.

#     Random Classifier: This method randomly assigns class labels to instances based on the distribution of classes in the training data.

#     Simple Heuristic Rules: These rules are based on domain knowledge or intuition. For example, in sentiment analysis, a simple rule could be to classify all reviews with the word "good" as positive and all reviews with the word "bad" as negative.

#     Naive Bayes Classifier: Although Naive Bayes is a simple probabilistic classifier, it often serves as a baseline due to its simplicity and sometimes surprisingly good performance, especially with text classification tasks.

#     Decision Trees: A single decision tree without any optimization or pruning can serve as a baseline. Decision trees are interpretable and easy to understand, making them suitable as baselines.

#     Logistic Regression: Logistic regression is a simple linear model commonly used for classification tasks. It's often used as a baseline due to its simplicity and interpretability.

# These baseline methods provide a starting point for evaluating more complex models. If a sophisticated model cannot outperform these simple approaches, it suggests either a problem with the model or the need for more data or feature engineering.