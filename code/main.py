from globals import *
from basic import basic
from num_sex import num_sex
from ThreeD_Scatter import ThreeD_Scatter
from clean import clean
from standardize import standardize
from islands import islands
from island_cofounding import island_cofounding
from species import species
from bill_len import bill_len

from bill_len_and_depth import bill_len_and_depth
from heatmap_of_numericals import heatmap_of_numericals
from pairwise_numericals import pairwise_numericals
from knn import knn

# the dataset can be found among these
# print(sns.get_dataset_names())

# get the pengium dataset
df=sns.load_dataset('penguins')

# define custom colors
custom_colors = {'Adelie': 'blue', 'Chinstrap': 'red', 'Gentoo': 'green'}

# show basic stuff about the penguins
# put baseline here
basic(df)

# plot numerical features against sex
num_sex(df)

# pairwise plot of the numerial variables
pairwise_numericals(df, custom_colors)

# clean the data and remove missing values
clean(df)

# standardize the numerical features
standardize(df)

# 3D scatter
ThreeD_Scatter(df, custom_colors)

# one hot encoding - only needed for methods that can only be numerical  
#  e.g. not needed for DTs, but it is for linear models and NNs 

# show the species on each of the islands where the penguins live
islands(df, custom_colors)

# is the island a cofounding factor in altering mass/size of pengiun?
# island_cofounding(df)

        # show the numbers of the species
        # species(df)

        # plot bill length for each species on each island
        # bill_len(df)

        # plot bill length and bill depth
        # bill_len_and_depth(df)

        # heatmap of numerical features
        # heatmap_of_numericals(df)

# knn
knn(df)
