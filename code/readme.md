## Tim Mulvaney - Question 1 Visualization and analysis of the Palmer penguin dataset

This directory contains the code to support the visualization, training and testing of the Palmer penguin dataset

The code is controlled by the file main.py, which just calls the other Python script in turn. 
After calls to the scripts to load and clean the data a number of processing options are available, 
each of which is executed by a single function call. 

The Python scripts that are to be executed in the current run just need to be 'commented in'. 
This allows specific processing to be selected for a particular run without having to execute superfluous code. 

The options available are as follows.
 - baseline classification for the penguins
 - plot of the numbers of each type of species
 - plot species' numerical features for each sex
 - plot the number of species on each of the islands where the penguins live
 - test whether island is a cofounding factor in the physical attributes of the penguins
 - produce pairwise plot of the numerical variables
 - standardize - use numerical feature training data to modify feature data to a mean of zero and standard deviation of unity
 - plot bill length for each species on each island
 - plot bill length against bill depth
 - produce a heatmap of the numerical data
 - determine feature importance
 - knn training and testing
 - random forest training and testing
 - perform an unusual and interesting combined visualization and analysis (CVA) that includes SVM training and testing
 - logic regression training and testing

