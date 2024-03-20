from globals import *

def baseline(local_df):
    
  # libraries to split into training/test sets, for knn and for f1 score
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


  # logistic regression baseline calculations
 
  # the sum of the accuracies from all the ogistic regression tests
  lg_accuracy = 0

  # the number of separate tests using ogistic regression classification (lg_max > 0)
  lg_max = 10

  # do random forest classiciation for rf_max random states (random_state=0 is avoided as its pseudo random, not fixed)
  for random_state in range (1,lg_max+1):
      
    # divide into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # choose a model
    model = LogisticRegression(random_state=random_state, max_iter=10000)

    # train the model
    model.fit(X_train, y_train)

    # make preditions using the test set
    y_pred = model.predict(X_test)

    # keep the sum of the accuracies so far
    lg_accuracy += accuracy_score(y_test, y_pred)

  # overall accuracy for valuating the model
  print(f"Logistic regression accuracy: {100*lg_accuracy/lg_max:.2f}%")



  # random forest baseline calculations
 
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

