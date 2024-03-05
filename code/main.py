from globals import *
from basic import *
from clean import *
from islands import *
from island_cofounding import *
from species import *
from bill_len import *
from mass_sex import *
from bill_len_and_depth import *
from heatmap_of_numericals import *
from pairwise_numericals import *
from knn import *

# the dataset can be found among these
# print(sns.get_dataset_names())

# get the pengium dataset
df=sns.load_dataset('penguins')

# show basic stuff about the penguins
# put baseline here
basic(df)

# clean the data and remove missing values
clean(df)

# show the islands where the penguins live
islands(df)

# is the island a cofounding factor in altering mass/size of pengiun?
island_cofounding(df)

# show the numbers of the species
# species(df)

# plot bill length for each species on each island
# bill_len(df)

# plot mass and sex
mass_sex(df)

# plot bill length and bill depth
# bill_len_and_depth(df)

# heatmap of numerical features
# heatmap_of_numericals(df)

# pairwise plot of the numerial variables
pairwise_numericals(df)

# knn
knn(df)
