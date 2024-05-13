from globals import * 

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# show some stuff about the penguins
def baseline(local_df):
  
  # Define features and target variable
  X = local_df.drop(columns=['species'])
  y = local_df['species']

  # Split dataset into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create a DummyClassifier with strategy 'most_frequent'
  baseline_model = DummyClassifier(strategy='most_frequent')

  # Fit the baseline model
  baseline_model.fit(X_train, y_train)

  # Evaluate the baseline model on the test set
  accuracy = baseline_model.score(X_test, y_test)
  print("Baseline accuracy using majority:", accuracy)

    

  # get a local copy of the df to manipulate
  df_copy = local_df.copy()

  # ecode categorical species column
  label_encoder = LabelEncoder()
  df_copy['species'] = label_encoder.fit_transform(df_copy['species'])
  df_copy['island'] = label_encoder.fit_transform(df_copy['island'])
  # df_copy['bill_length_mm'] = label_encoder.fit_transform(df_copy['bill_length_mm'])
  # df_copy['bill_depth_mm'] = label_encoder.fit_transform(df_copy['bill_depth_mm'])
  # df_copy['flipper_length_mm'] = label_encoder.fit_transform(df_copy['flipper_length_mm'])
  # df_copy['body_mass_g'] = label_encoder.fit_transform(df_copy['body_mass_g'])
  df_copy['sex'] = label_encoder.fit_transform(df_copy['sex'])

  for feature in df_copy.columns.tolist():
    # define baseline feature and target variable
    X = df_copy[[feature]]    
    y = df_copy['species']

    # split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create and train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # predict species on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Baseline accuracy using {feature}: {accuracy}")




# A baseline method for classification is a simple model or technique used as a point of reference for comparing the performance of more complex models. It serves as a benchmark against which the performance of more sophisticated models can be evaluated. Baseline methods are typically straightforward and easy to implement, providing a basic level of performance that any more advanced model should aim to surpass.

# Common baseline methods for classification include:

#     Majority Class Classifier: This method simply predicts the most frequent class in the training data for every instance in the test data. It's useful for imbalanced datasets where one class dominates.

#     Random Classifier: This method randomly assigns class labels to instances based on the distribution of classes in the training data.

#     Simple Heuristic Rules: These rules are based on domain knowledge or intuition. For example, in sentiment analysis, a simple rule could be to classify all reviews with the word "good" as positive and all reviews with the word "bad" as negative.

#     Naive Bayes Classifier: Although Naive Bayes is a simple probabilistic classifier, it often serves as a baseline due to its simplicity and sometimes surprisingly good performance, especially with text classification tasks.

#     Decision Trees: A single decision tree without any optimization or pruning can serve as a baseline. Decision trees are interpretable and easy to understand, making them suitable as baselines.

#     Logistic Regression: Logistic regression is a simple linear model commonly used for classification tasks. It's often used as a baseline due to its simplicity and interpretability.

# These baseline methods provide a starting point for evaluating more complex models. If a sophisticated model cannot outperform these simple approaches, it suggests either a problem with the model or the need for more data or feature engineering.