from globals import *
from basic import basic
from clean import clean
from islands import islands
from island_cofounding import island_cofounding
from species import species
from bill_len import bill_len
from num_sex import num_sex
from bill_len_and_depth import bill_len_and_depth
from heatmap_of_numericals import heatmap_of_numericals
from pairwise_numericals import pairwise_numericals
from knn import knn

# the dataset can be found among these
# print(sns.get_dataset_names())

# get the pengium dataset
df=sns.load_dataset('penguins')

# show basic stuff about the penguins
basic(df)


# plot numerical features against sex
num_sex(df)

# clean the data and remove missing values
clean(df)

# show the islands where the penguins live
# islands(df)

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

# pairwise plot of the numerial variables
# pairwise_numericals(df)

# knn
knn(df)
