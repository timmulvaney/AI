## Tim Mulvaney - Question 1 Visualization and analysis of the Palmer penguin dataset

This repository contains the code and documentation to support the visualization and analysis of the 
Palmer penguin dataset

### Files in the github repository

The githb repository containing the project can be found at https://github.com/timmulvaney/AI

The following directories can be found in the repository.
- brief - the coursework specification
- code - the Python implementation of the data cleaning, encoding. standardization, as well as the AI training and testing
- kmeans - the code for kmeans was originally written as a jupyter notebook, but a Python version was written for easier integration with the remaining code
- penguin reference - links to other work on Penguin data analysis
- report - the separate report on ....?

### Report

The report is witten in LaTeX and is found in the file report.tex. A separate BibTex file report.bib contains the
references. The report is compiled to pdf output using the bash script 'gen', for example ./gen report.

### Code

The code is controlled by the file main.py, which just calls the other Python script in turn. After calls to the scripts to load and clean the data a number of processing options are available, each of which is exectuted by a single function call. The Python scripts that are to be executed in the current run just need to be commented out. This allows specific processing to be selected for a particular run without having to execute the superflous code. 

The options available are as follows.
 - baseline classification for the penguins
 - plot of the numbers of each type of species
 - plot species' numerical features for each sex
 - plot the number of species on each of the islands where the penguins live
 - test whether island is a cofounding factor in the physical attributes of the pengiuns
 - produce pairwise plot of the numerial variables
 - standardize - use numerical feature training data to modify feature data to a mean of zero and standard deviation of unity
 - produce one-hot encoded versions of the dataset  
 - plot bill length for each species on each island
 - plot bill length against bill depth
 - produce a heatmap of the numerical data
 - determine feature importance
 - knn training and testing
 - random forest training and testing

k means
in separate jupyter notebook

 - perform an unusual and interesting combined visualization and analysis (CVA) that includes SVMs
 - logistic regression analysis

