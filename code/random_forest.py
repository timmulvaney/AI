from globals import *

def random_forest(local_df):
    
  # libraries to split into training/test sets, for logistic regression, random forest and accuracy
  from sklearn.model_selection import cross_val_score, train_test_split
  from sklearn.preprocessing import LabelEncoder
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score

  # get a local copy of the df to manipulate
  df_copy = local_df.copy()

  # ecode categorical species column
  label_encoder = LabelEncoder()
  df_copy['species'] = label_encoder.fit_transform(df_copy['species'])
  df_copy['island'] = label_encoder.fit_transform(df_copy['island'])
  df_copy['sex'] = label_encoder.fit_transform(df_copy['sex'])

  df_copy.drop(columns=['island'], inplace =True)
  # df_copy.drop(columns=['sex'], inplace =True)
  # df_copy.drop(columns=['bill_length_mm'], inplace =True)
  # df_copy.drop(columns=['bill_depth_mm'], inplace =True)
  # df_copy.drop(columns=['flipper_length_mm'], inplace =True)
  df_copy.drop(columns=['body_mass_g'], inplace =True)
  
  # separate features and target
  X = df_copy.drop('species', axis=1)
  y = df_copy['species']



  #
  # random forest calculations
  #
  
  # the sum of the accuracies from all the random forest tests
  rf_accuracy = 0

  # the number of separate tests using random forest classification (rf_max > 0)
  rf_max = 10

  # do random forest classiciation for rf_max random states (random_state=0 is avoided as its pseudo random, not fixed)
  for random_state in range (1,rf_max+1):
      
    # divide into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # choose a model
    model = RandomForestClassifier(random_state=random_state)

    # train the model
    model.fit(X_train, y_train)

    # make preditions using the test set
    y_pred = model.predict(X_test)

    # keep the sum of the accuracies so far
    rf_accuracy += accuracy_score(y_test, y_pred)

  # overall accuracy for valuating the model
  print(f"Random forest accuracy: {100*rf_accuracy/rf_max:.2f}%")


  #
  # random forest hyperparameter calculations
  #

  # the sum of the accuracies from all the random forest tests
  rf_accuracy = 0

  # the number of separate tests using random forest classification (rf_max > 0)
  rf_max = 10


  # do random forest classiciation for rf_max random states (random_state=0 is avoided as its pseudo random, not fixed)
  # ADD grid search
  from sklearn.model_selection import train_test_split, GridSearchCV
  
  # parameter grid to search
  param_grid = {
    'n_estimators': [10, 25, 50],  # Number of trees in the forest
    'max_depth': [None, 10, 20],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'criterion': ['gini', 'entropy']  # function to measure quality of split
  }

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

    # Evaluate the best model on test data
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy on random set {random_state}: {test_accuracy}")

    # keep the sum of the accuracies so far
    rf_accuracy += accuracy_score(y_test, y_pred)

  # overall accuracy for valuating the model
  print(f"Random forest accuracy: {100*rf_accuracy/rf_max:.2f}%")
