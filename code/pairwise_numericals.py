from globals import * 

# def pairwise_numericals(local_df, custom_colors):

#   # Multiple Bivariate distribution
#   # Linear relationships can be seen in most of the plots 
#   sns.set_style(style='white')

#   # Define the order of species and corresponding colors
#   hue_order = ['Adelie', 'Chinstrap', 'Gentoo']
#   # palette = {'Adelie': '#01193f', 'Chinstrap': '#6baddf', 'Gentoo': '#d2b486'}

#   # Plot with specified hue order and palette - ss.pairplot ignores non-numerical features
#   sns.pairplot(data=local_df, hue='species', hue_order=hue_order, palette=custom_colors)

#   # For each species, the joint distribution of the pair plots is approximately oval in shape.
#   # Where the main axes of the ovals appears ia along a diagonal this indicates a correlation between features
#   plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

def pairwise_numericals(local_df, custom_colors):
  # Multiple Bivariate distribution
  # Linear relationships can be seen in most of the plots 
  sns.set_style(style='white')

  # Define the order of species and corresponding colors
  hue_order = ['Adelie', 'Chinstrap', 'Gentoo']

  # Plot with specified hue order and palette - ss.pairplot ignores non-numerical features
  g = sns.pairplot(data=local_df, hue='species', hue_order=hue_order, palette=custom_colors, corner=True)

  # to change the legends location
  handles = g._legend_data.values()
  labels = g._legend_data.keys()
  g.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=1)
  g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3)
  g.fig.legend(handles=handles, labels=labels, loc='upper left', ncol=1)
  g.fig.subplots_adjust(top=0.92, bottom=0.08)
  plt.show()



