from globals import *

def knn():
    
  # libraries to split into training/test sets, for knn and for f1 score
  from sklearn.model_selection import cross_val_score, train_test_split
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import f1_score

  # map categorical species (target) to integers for knn classification 
  df['species'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
  df['island'] = pd.Categorical(df['island']).codes
  df['sex'] = pd.Categorical(df['sex']).codes

  # separate features and target
  X = df.drop('species', axis=1)
  y = df['species']

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


    # need to try different random states for f1????