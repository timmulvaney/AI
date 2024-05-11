from globals import *

# libraries to split into training/test sets, for logistic regression, random forest and accuracy
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# perform random forest training and test using the dataframe passed
def random_forest(local_df):
    
  # get a local copy of the df to manipulate
  df_copy = local_df.copy()

  # encode categorical columns
  label_encoder = LabelEncoder()
  df_copy['species'] = label_encoder.fit_transform(df_copy['species'])
  df_copy['island'] = label_encoder.fit_transform(df_copy['island'])
  df_copy['sex'] = label_encoder.fit_transform(df_copy['sex'])

  # comment out the following if the corresponding feature is to be dropped
  # df_copy.drop(columns=['island'], inplace =True)
  # df_copy.drop(columns=['sex'], inplace =True)
  # df_copy.drop(columns=['bill_length_mm'], inplace =True)
  # df_copy.drop(columns=['bill_depth_mm'], inplace =True)
  # df_copy.drop(columns=['flipper_length_mm'], inplace =True)
  # df_copy.drop(columns=['body_mass_g'], inplace =True)
  
  # separate features and target
  X = df_copy.drop('species', axis=1)
  y = df_copy['species']
  
  # parameter grid to search
  param_grid = {
    'n_estimators': [10, 15, 20, 25],  # Number of trees in the forest
    'max_depth': [None, 10, 20],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'criterion': ['gini', 'entropy']  # function to measure quality of split
  }

  # find best metaparameters
  train_random_forest(param_grid, X, y)
  
  # this is the set of best parameters - this needs to be filled in manually at present
  param_grid = {
    'n_estimators': [10],  # Number of trees in the forest
    'max_depth': [None],  # Maximum depth of the trees
    'min_samples_split': [2],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1],  # Minimum number of samples required at each leaf node
    'criterion': ['gini']  # function to measure quality of split
  }

  # use best set of metaparameters to get the test set result - this needs to be filled in manually at present
  train_random_forest(param_grid, X, y)  


# perform the random forest training and testing
def train_random_forest(param_grid, X, y):

  # the sum of the accuracies from all the random forest tests
  rf_accuracy = 0

  # the number of separate tests using random forest classification (rf_max > 0)
  rf_max = 100

  # initialize counts of best parameters
  best_params_count = {}

  for random_state in range (1,rf_max+1):
      
    # divide into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # choose a model
    model = RandomForestClassifier(random_state=random_state)

    # grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # make preditions using the test set
    #  = model.predict(X_test)

    # Print the best parameters found
    print("Best Parameters:", grid_search.best_params_)

    # make hashable
    best_params_tuple = tuple(sorted(grid_search.best_params_.items()))
    
    # increment the count for the current best parameters
    best_params_count[best_params_tuple] = best_params_count.get(best_params_tuple, 0) + 1

    # Evaluate the best model on test data
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy on random set {random_state}: {test_accuracy}")

    # keep the sum of the accuracies so far
    rf_accuracy += accuracy_score(y_test, y_pred)

    if (random_state==1):
      rf_best_accuracy = test_accuracy
      rf_worst_accuracy = test_accuracy
    else:
      if(test_accuracy>rf_best_accuracy):
        rf_best_accuracy = test_accuracy
      if(test_accuracy<rf_worst_accuracy):
        rf_worst_accuracy = test_accuracy

  # print count of best parameters
  for params, count in best_params_count.items():
    print("Parameters:", dict(params), "Count:", count)

  # overall accuracy for evaluating the model
  print(f"Random forest accuracy: {100*rf_accuracy/rf_max:.2f}%")
  print(f"Random forest best accuracy: {100*rf_best_accuracy:.2f}%")  
  print(f"Random forest worst accuracy: {100*rf_worst_accuracy:.2f}%") 
