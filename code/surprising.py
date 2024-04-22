from globals import * 

# needed for dictionary
import ast 

# extra font properties for bold in the legend
from matplotlib.font_manager import FontProperties

def surprising(local_df, custom_colors):

  # get a copy of the passed dataframe
  plotted_df = local_df.copy()

  # Define variables for the scatter plot
  x = 'bill_depth_mm'
  y = 'flipper_length_mm'
  size = 'sex'

  # Create the scatter plot to separate Gentoo 
  plt.figure(figsize=(10, 7))
  g = sns.scatterplot(data=plotted_df, x=x, y=y, size=size, hue='species', palette=custom_colors, sizes=(50, 120), alpha=0.8)
  # adjust legend
  g.legend().remove()
  # g.figure.legend(loc=(0.75,0.6), ncol=1, fontsize=12)

  # Adjust legend font properties to make titles bold
  boldfont = FontProperties()
  boldfont.set_weight('bold')
  boldfont.set_size(18)
  normfont = FontProperties()
  normfont.set_size(18)

  legend = plt.legend()
  for i, text in enumerate(legend.get_texts()):
    if text.get_text() in ["species", "sex", "island"]:
      text.set_font_properties(boldfont)       
    else:
      text.set_font_properties(normfont)      

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
  plt.xlabel('bill depth (mm)', fontsize=20)
  plt.xticks(fontsize=15)
  plt.ylabel('flipper length (mm)', fontsize=20)
  plt.yticks(fontsize=15)
  plt.title('Penguin Morphological Measurements', fontsize=18)

  plt.show()



  # get a copy of the df we can modify
  svn_df = plotted_df.copy()

  # Gentoo has already been done
  svn_df = svn_df[svn_df['species'] != 'Gentoo']

  # define whether the plot is to be produced for Male, Female or Both sexes - comment out all but one
  # sex_plot = "male"
  sex_plot = "female"
  # sex_plot = "both male and female"

  # if single sex, keep the one we want
  if (sex_plot == "male"):
    svn_df = svn_df[svn_df['sex'] != 'Female']
  if (sex_plot == "female"):
    svn_df = svn_df[svn_df['sex'] != 'Male']

  # if single sex, we include the island (and drop sexes), for combined sexes we need to keep the sexes (and drop the island)
  svn_df['species'] = svn_df['species'].map({'Adelie': 0, 'Chinstrap': 1})
  if ((sex_plot == "male") or (sex_plot == "female")):
    svn_df['island'] = svn_df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen': 2})
    svn_df.drop(columns=['sex'], inplace =True) 
  else:
    svn_df['sex'] = svn_df['sex'].map({'Male': 0, 'Female': 1})
    svn_df.drop(columns=['island'], inplace =True) 

  # choose two of the numerical features and drop the rest
  numerical_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
  x = 'bill_length_mm'
  x_label = 'bill length (mm)'
  y = 'bill_depth_mm'
  y_label = 'bill depth (mm)'
  # y = 'flipper_length_mm'
  # y_label = 'flipper length (mm)'
  # y = 'body_mass_g'
  # y_label = 'body mass (g)'

  # drop the columns we don't need
  for num_col in numerical_columns:
    if ((num_col != x) and (num_col != y)):
      svn_df.drop(columns=[num_col], inplace =True)  

  # drop the species from the training data (keep for target) and drop island from training for single sex
  if ((sex_plot == "male") or (sex_plot == "female")):
    X_in = svn_df.drop(columns = ['species','island'], axis=1)
  else:
    X_in = svn_df.drop('species', axis=1)

  # X_in = svn_df.drop(['species','island'], axis=1)

  # the target is the species
  y_in = svn_df['species']

  # do the training for the line of best fit
  X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.2, random_state=10)
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y_train)

  # get the coefficients and parameters of the separating hyperplane
  w = clf.coef_[0]
  b = clf.intercept_[0]
  slope = -w[0]/w[1]
  intercept = -b/w[1]
  
  # create the scatter plot  
  plt.figure(figsize=(10, 7))
  # x = 'bill_length_mm'
  # y = 'bill_depth_mm'   # -------- this best???

  if ((sex_plot == "male") or (sex_plot == "female")):
    size='island'
    temp_df = plotted_df[plotted_df['species'] != 'Gentoo']
    g = sns.scatterplot(data=temp_df, x=x, y=y, size=size, hue='species', sizes=(40,150), palette=custom_colors, alpha=0.8)
    print("temp_df")
    print(temp_df)
  else:
    size='sex'
    temp_df = plotted_df[plotted_df['species'] != 'Gentoo']
    g = sns.scatterplot(data=temp_df, x=x, y=y, size=size, hue='species', sizes=(40,150), palette=custom_colors, alpha=0.8)
    print("temp_df")
    print(temp_df)
  
  # adjust legend
  g.legend().remove()
  # g.figure.legend(loc=(0.75,0.6), ncol=1, fontsize=12)

  # Adjust legend font properties to make titles bold
  boldfont = FontProperties()
  boldfont.set_weight('bold')
  boldfont.set_size(18)
  normfont = FontProperties()
  normfont.set_size(18)

  legend = plt.legend()
  for i, text in enumerate(legend.get_texts()):
    if text.get_text() in ["species", "sex", "island"]:
      text.set_font_properties(boldfont)       
    else:
      text.set_font_properties(normfont)    
      
  # add the svm line 
  y_line = [min(X_train[y].tolist()), max(X_train[y].tolist())]
  x_line = [(y_line[0] - intercept)/slope, (y_line[1] - intercept)/slope]
  plt.plot(x_line, y_line, color='black', label='Line')
 
  # add labels and title before plotting
  # plt.xlabel(x_label)
  # plt.ylabel(y_label)
  # plt.title(f'Physical differences of {sex_plot} Adelie and Chinstrap species')
  # plt.show()

  plt.xlabel(x_label, fontsize=20)
  plt.xticks(fontsize=15)
  plt.ylabel(y_label, fontsize=20)
  plt.yticks(fontsize=15)
  plt.title(f'Physical differences of {sex_plot} Adelie and Chinstrap species')
  plt.show()

 
  # Standardize features by removing the mean and scaling to unit variance
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  
  # the sum of the accuracies from all the logistic regressiont tests
  svm_accuracy = 0

  # number of random states to try
  svm_max = 10
  
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


