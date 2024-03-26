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

  # Define features (island) and target variable (species)
  X = df_copy[['island']]
  y = df_copy['species']

  # Split dataset into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create and train a logistic regression model
  model = LogisticRegression(max_iter=1000)
  model.fit(X_train, y_train)

  # Predict species on the test set
  y_pred = model.predict(X_test)

  # Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)
  print("Baseline accuracy using island:", accuracy)
