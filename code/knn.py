from globals import *

from sklearn.metrics import accuracy_score

def knn(local_df):
    
  # libraries to split into training/test sets, for knn and for f1 score
  from sklearn.model_selection import cross_val_score, train_test_split
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import f1_score
  from sklearn.metrics import confusion_matrix

  # get a local copy of the df to manipulate
  copy_df = local_df.copy()

  # map categorical species (target) to integers for knn classification 
  copy_df['species'] = copy_df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
  copy_df['island'] = pd.Categorical(copy_df['island']).codes
  copy_df['sex'] = pd.Categorical(copy_df['sex']).codes
  
  # copy_df.drop(columns=['island'], inplace =True)
  # copy_df.drop(columns=['sex'], inplace =True)
  # copy_df.drop(columns=['bill_length_mm'], inplace =True)
  # copy_df.drop(columns=['bill_depth_mm'], inplace =True)
  copy_df.drop(columns=['flipper_length_mm'], inplace =True)
  copy_df.drop(columns=['body_mass_g'], inplace =True)

  # separate features and target
  X = copy_df.drop('species', axis=1)
  y = copy_df['species']

  # divide into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # try 1 to 10 nearest neighbours
  for k in range(1,10):

    # initialize kNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # perform cross-validation
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

    # train the KNN classifier
    knn.fit(X_train, y_train)
    
    # find predicted outputs for the test set
    y_pred = knn.predict(X_test)

    # calculate the F1-score
    f1 = f1_score(y_test, y_pred, average='weighted')

    # print cross-validation scores
    # print("cross-validation values for k =", k, ":", cv_scores)
    print("mean cross-validation accuracy for k =", k, ":", np.mean(cv_scores))
    print("f1 score for k =", k, ":", f1)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for k={k}: {accuracy}')


  # lots of hyperparams to play with 
  # need to try different random states for f1????
  # accuracy????
    
  # 
  # knn grid search for metaparemters

    
  # the sum of the accuracies from all the random forest tests
  knn_accuracy = 0

  # the number of separate tests using random forest classification (knn_max > 0)
  knn_max = 100

  # do logistic regression for knn_max random states (random_state=0 is avoided as its pseudo random, not fixed)
  # ADD grid search
  from sklearn.model_selection import train_test_split, GridSearchCV
  
  # parameter grid to search
  param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 8, 10],  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'metric': ['manhattan', 'euclidean']  # Power parameter for the Minkowski distance metric
  }


  # train and find accuracy for knn_max random states
  for random_state in range (1,knn_max+1):
      
    # divide into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)




    # choose a model
    model = KNeighborsClassifier()

    # grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best parameters found
    print("Best Parameters:", grid_search.best_params_)

    # Evaluate the best model on test data
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy on random set {random_state}: {test_accuracy}")

    # keep the sum of the accuracies so far
    knn_accuracy += accuracy_score(y_test, y_pred)

  # overall accuracy for evaluating the model
  print(f"knn accuracy: {100*knn_accuracy/knn_max:.2f}%")

  cm = confusion_matrix(y_test, y_pred)
  print("Predicted on x, True on y: 'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2")
  print(cm)