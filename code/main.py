from globals import *
from basic import basic
# from clean import clean
from clean import clean
from baseline import baseline
from logistic_regression import logistic_regression
from species import species
from num_sex import num_sex
from islands import islands
from island_confounding import island_confounding
from pairwise_numericals import pairwise_numericals
from standardize import standardize
from surprising import surprising
from bill_len import bill_len
from bill_len_and_depth import bill_len_and_depth
from heatmap_of_numericals import heatmap_of_numericals
from feature_importance import feature_importance
from random_forest import random_forest
from knn import knn


# control the operation of the program

# the dataset can be found among these
# print(sns.get_dataset_names())

# get the pengium dataset
df=sns.load_dataset('penguins')

# define custom colors
custom_colors = {'Adelie': 'darkorange', 'Chinstrap': 'mediumorchid', 'Gentoo': 'mediumseagreen'}

# show basic stuff about the penguins
basic(df)

# consider balanced/unbalanced
# probs just argue that this is not a problem

# clean the data and remove missing values
clean_df = clean(df)

# Specify the file path and save the CSV file
# output_file = 'penguin_cleaned.csv'
# clean_df.to_csv(output_file, index=False)  # Set index=False to exclude the DataFrame index from the output

# baseline classification for the penguins
# baseline(clean_df)

# show the numbers of the species
# species(clean_df, custom_colors)

# plot numerical features against sex
# num_sex(clean_df, custom_colors)

# show the species on each of the islands where the penguins live
# islands(clean_df, custom_colors)

# is the island a cofounding factor in altering mass/size of pengiun?
# island_confounding(clean_df)

# pairwise plot of the numerial variables
# pairwise_numericals(clean_df, custom_colors)

# get a version of the df with the numerical features to have a mean of zero and standard deviation of unity
stand_df = standardize(clean_df)

# one hot encoding - only needed for methods that can only be numerical  
#  e.g. not needed for DTs, but it is for linear models and NNs 

# plot bill length for each species on each island
# bill_len(df)

# plot bill length and bill depth
# bill_len_and_depth(df)

# heatmap of numerical features
# heatmap_of_numericals(clean_df)

# determine feature importance
feature_importance(stand_df)

# knn analysis
# knn(clean_df)

# random forest analysis
# random_forest(clean_df)

# k means
# in separate jupyter notebook

# unusual and interesting mix of visualization and classification? 
surprising(clean_df, custom_colors)

# logistic regression analysis
# logistic_regression(clean_df)