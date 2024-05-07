from globals import * 
from sklearn.calibration import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

def feature_importance(local_df):

  # get a copy of the df to play with
  temp_df = local_df

  # encode 'species' 'island' and 'sex' features
  label_encoder_species = LabelEncoder()
  temp_df['species_encoded'] = label_encoder_species.fit_transform(temp_df['species'])
  label_encoder_island = LabelEncoder()
  temp_df['island_encoded'] = label_encoder_island.fit_transform(temp_df['island'])
  label_encoder_sex = LabelEncoder()
  temp_df['sex_encoded'] = label_encoder_sex.fit_transform(temp_df['sex'])

  # split data into features (X) and target variable (y)
  X = temp_df.drop(['species','island','sex','species_encoded'], axis=1) 
  y = temp_df['species_encoded']

  print(X)
  print(y)

  ############ comment out one of the following classifiers

  # Train XGBoost classifier
  method = 'XGBoost'
  xgb_classifier = xgb.XGBClassifier(random_state=42)
  xgb_classifier.fit(X, y)
  feature_importances = xgb_classifier.feature_importances_

  # # Train Random Forest classifier
  # method = 'Random Forest'
  # rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
  # rf_classifier.fit(X, y)
  # feature_importances = rf_classifier.feature_importances_

  # Create DataFrame of feature importances
  feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
  feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

  # Plot feature importances
  plt.figure(figsize=(10, 8))
  plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='lightgreen')
  plt.xlabel('Importance')
  plt.ylabel('Feature')
  plt.title('Feature Importance Scores (' + method + ')')
  plt.show()

  print("Feature Importance Scores (XGBoost):")
  print(feature_importance_df)

