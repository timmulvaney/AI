from globals import * 

# needed for dictionary
import ast 

def ThreeD_Scatter(local_df, custom_colors):

  # get a copy of the passed dataframe
  plotted_df = local_df.copy()

  # Define variables for the scatter plot
  x = 'bill_depth_mm'
  y = 'flipper_length_mm'
  size = 'bill_length_mm'

  # Create the scatter plot to separate Gentoo 
  plt.figure(figsize=(10, 7))
  sns.scatterplot(data=plotted_df, x=x, y=y, size=size, hue='species', palette=custom_colors, sizes=(20, 200), alpha=0.8)

  # train svm module
  from sklearn import svm
  from sklearn.model_selection import cross_val_score, train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.svm import SVC
  from sklearn.metrics import accuracy_score

  svn_df = plotted_df.copy()
  svn_df['species'] = svn_df['species'].map({'Adelie': 0, 'Chinstrap': 0, 'Gentoo': 1})
  svn_df.drop(columns=['island'], inplace =True)
  svn_df.drop(columns=['sex'], inplace =True)
  svn_df.drop(columns=['bill_length_mm'], inplace =True)
  # svn_df.drop(columns=['bill_depth_mm'], inplace =True)
  # svn_df.drop(columns=['flipper_length_mm'], inplace =True)
  svn_df.drop(columns=['body_mass_g'], inplace =True)

  # separate features and target
  print(svn_df.head(11))
  X_in = svn_df.drop('species', axis=1)
  y_in = svn_df['species']
  X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.2, random_state=42)
  print(X_train)
  print(y_train)

  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y_train)

  # Get the coefficients of the separating hyperplane
  w = clf.coef_[0]
  b = clf.intercept_[0]

  # Equation of the separating line
  slope = -w[0]/w[1]
  intercept = -b/w[1]

  print("slope", slope)
  print("intercept", intercept) 

  # my guess of a line from the graph - define points for the line and plot
  x_line = [13, 21]
  y_line = [180, 240]
  # plt.plot(x_line, y_line, color='yellow', label='Line')

  # svn line 
  x_line = [13, 21]
  y_line = [slope*x_line[0] + intercept, slope*x_line[1] + intercept]
  # x_line = [180,240]
  # y_line = [slope*x_line[0] + intercept, slope*x_line[1] + intercept]
  print("y_line", y_line)
  plt.plot(x_line, y_line, color='black', label='Line')

  # Standardize features by removing the mean and scaling to unit variance
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # Initialize SVM classifier
  svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

  # Train the SVM classifier
  svm_classifier.fit(X_train, y_train)

  # Predict the classes for test data
  y_pred = svm_classifier.predict(X_test)

  # Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)
  print(f'Accuracy: {accuracy}')

  # add labels and title before plotting
  plt.xlabel('Bill Depth (mm)')
  plt.ylabel('Flipper Length (mm)')
  plt.title('3D Scatter Plot of Penguin Morphological Measurements')
  plt.show()



  # define variables for the scatter plot
  x = 'bill_length_mm'
  # y = 'flipper_length_mm'
  y = 'bill_depth_mm'   # -------- this best???
  # y = 'body_mass_g'
  size='sex'

  svn_df = plotted_df.copy()
  svn_df = svn_df[svn_df['species'] != 'Gentoo']
  # svn_df['species'] = svn_df['species'].map({'Adelie': 0, 'Chinstrap': 1})
  svn_df.drop(columns=['island'], inplace =True)
  # svn_df.drop(columns=['sex'], inplace =True)
  svn_df = svn_df[svn_df['sex'] != 'Male']
  svn_df['sex'] = svn_df['sex'].map({'Male': 0, 'Female': 1})
  # svn_df.drop(columns=['bill_length_mm'], inplace =True)
  # svn_df.drop(columns=['bill_depth_mm'], inplace =True)
  svn_df.drop(columns=['flipper_length_mm'], inplace =True)
  svn_df.drop(columns=['body_mass_g'], inplace =True)

  # Create the scatter plot  
  plt.figure(figsize=(10, 7))
  sns.scatterplot(data=svn_df, x=x, y=y, size=size, hue='species', sizes =(150,40), palette=custom_colors, alpha=0.8)
 
  # use svn to separate targets
  X_in = svn_df.drop('species', axis=1)
  y_in = svn_df['species']
  X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.2, random_state=4)
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y_train)

  # get the coefficients and parameters of the separating hyperplane
  w = clf.coef_[0]
  b = clf.intercept_[0]
  slope = -w[0]/w[1]
  intercept = -b/w[1]
  x_line = [40,44]   # 
  # x_line = plt.xlim()
  y_line = [slope*x_line[0] + intercept, slope*x_line[1] + intercept]

  # draw the line 
  print("y_line", y_line)
  plt.plot(x_line, y_line, color='black', label='Line')

  # Standardize features by removing the mean and scaling to unit variance
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  
  # the sum of the accuracies from all the logistic regressiont tests
  svm_accuracy = 0

  # number of random states to try
  svm_max = 100
  
  # train and find accuracy for svn_max random states
  for random_state in range (1,svm_max+1):
    
    # divide into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.2, random_state=random_state)
    
    # Initialize SVM classifier
    svm_classifier = SVC(kernel='linear', C=1.0, random_state=random_state)

    # Train the SVM classifier
    svm_classifier.fit(X_train, y_train)

    # Predict the classes for test data
    y_pred = svm_classifier.predict(X_test)

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy on random set {random_state}: {test_accuracy}")
 
    # keep the sum of the accuracies so far
    svm_accuracy += accuracy_score(y_test, y_pred)

  # overall accuracy for evaluating the model
  print(f"svm accuracy: {100*svm_accuracy/svm_max:.2f}%")

  # add labels and title before plotting
  plt.xlabel('Bill Length (mm)')
  plt.ylabel('Bill Depth (mm)')
  plt.title('3D Scatter Plot of Penguin Morphological Measurements')
  plt.show()
