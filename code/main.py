from globals import *
from basic import basic
from clean import clean
from baseline import baseline
from species import species
from num_sex import num_sex
from islands import islands
from island_cofounding import island_cofounding
from pairwise_numericals import pairwise_numericals
from standardize import standardize
from ThreeD_Scatter import ThreeD_Scatter
from bill_len import bill_len
from bill_len_and_depth import bill_len_and_depth
from heatmap_of_numericals import heatmap_of_numericals

from knn import knn

# the dataset can be found among these
# print(sns.get_dataset_names())

# get the pengium dataset
df=sns.load_dataset('penguins')

# define custom colors
custom_colors = {'Adelie': 'blue', 'Chinstrap': 'red', 'Gentoo': 'green'}

# show basic stuff about the penguins
basic(df)

# consider balanced/unbalanced
#  probs just argue that this is okay now

# clean the data and remove missing values
clean_df = clean(df)

# do some baseline analysis of the data
baseline(clean_df)

# show the numbers of the species
species(clean_df, custom_colors)

# put baseline here
# is this just majority 

# plot numerical features against sex
# num_sex(clean_df)

# show the species on each of the islands where the penguins live
# islands(clean_df, custom_colors)

# is the island a cofounding factor in altering mass/size of pengiun?
# island_cofounding(clean_df)

# pairwise plot of the numerial variables
# pairwise_numericals(clean_df, custom_colors)

# get a version of the df with the numerical features to have a mean of zero and standard deviation of unity
# stand_df = standardize(df)

# 3D scatter
ThreeD_Scatter(clean_df, custom_colors)

# one hot encoding - only needed for methods that can only be numerical  
#  e.g. not needed for DTs, but it is for linear models and NNs 


        # plot bill length for each species on each island
        # bill_len(df)

        # plot bill length and bill depth
        # bill_len_and_depth(df)

        # heatmap of numerical features
        # heatmap_of_numericals(df)

# knn
knn(clean_df)
