from globals import *

# sklean libraries for knn and training
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# perform knn training and test using the dataframe passed
def knn(local_df):
    
  # get a local copy of the df to manipulate
  copy_df = local_df.copy()

  # map categorical species (target) to integers for knn classification 
  copy_df['species'] = copy_df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
  copy_df['island'] = pd.Categorical(copy_df['island']).codes
  copy_df['sex'] = pd.Categorical(copy_df['sex']).codes
  
  # comment out the following if the corresponding feature is to be dropped
  # copy_df.drop(columns=['island'], inplace =True)
  # copy_df.drop(columns=['sex'], inplace =True)
  # copy_df.drop(columns=['bill_length_mm'], inplace =True)
  # copy_df.drop(columns=['bill_depth_mm'], inplace =True)
  # copy_df.drop(columns=['flipper_length_mm'], inplace =True)
  # copy_df.drop(columns=['body_mass_g'], inplace =True)

  # separate features and target
  X = copy_df.drop('species', axis=1)
  y = copy_df['species']
  
  # parameter grid of metaparameters to search
  param_grid_train = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 8, 10],  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'metric': ['manhattan', 'euclidean']  # Power parameter for the Minkowski distance metric
  }

  # find best metaparameters
  train_knn(param_grid_train, X, y)

  # this is the set of best parameters - this needs to be filled in manually at present
  param_grid_test = {
    'n_neighbors': [1],  # Number of neighbors to consider
    'weights': ['uniform'],  # Weight function used in prediction
    'metric': ['manhattan']  # Power parameter for the Minkowski distance metric
  }
  
  # use best set of metaparameters to get the test set result - this needs to be filled in manually at present
  train_knn(param_grid_test, X, y)



# perform the knn training and testing
def train_knn(param_grid, X, y):

  # the sum of the accuracies from all the knn tests
  knn_accuracy = 0

  # the number of separate tests using random forest classification (knn_max > 0)
  knn_max = 100

  # initialize counts of best parameters
  best_params_count = {}

  # train and find accuracy for knn_max random states
  for random_state in range (1,knn_max+1):
      
    # divide into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # standardize features by removing the mean and scaling to unit variance using training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  # apply standardization to test data

    # knn model
    model = KNeighborsClassifier()

    # grid search with cross-validation, cv is the number of folds
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # print the best parameters found for this training/validation data
    print("Best Parameters:", grid_search.best_params_)

    # make hashable
    best_params_tuple = tuple(sorted(grid_search.best_params_.items()))
    
    # increment the count for the current best parameters
    best_params_count[best_params_tuple] = best_params_count.get(best_params_tuple, 0) + 1

    # details for test data of best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy on random set {random_state}: {test_accuracy}")

    # keep the sum of the accuracies so far
    knn_accuracy += accuracy_score(y_test, y_pred)

  # print count of best parameters
  for params, count in best_params_count.items():
    print("Parameters:", dict(params), "Count:", count)

  # overall accuracy for evaluating the model
  print(f"knn accuracy: {100*knn_accuracy/knn_max:.2f}%")