from globals import * 

# def pairwise_numericals(local_df, custom_colors):

  # # Multiple Bivariate distribution
  # # Linear relationships can be seen in most of the plots 
  # sns.set_style(style='white')

  # # Define the order of species and corresponding colors
  # hue_order = ['Adelie', 'Chinstrap', 'Gentoo']
  # # palette = {'Adelie': '#01193f', 'Chinstrap': '#6baddf', 'Gentoo': '#d2b486'}

  # # Plot with specified hue order and palette - ss.pairplot ignores non-numerical features
  # sns.pairplot(data=local_df, hue='species', hue_order=hue_order, palette=custom_colors)

  # # For each species, the joint distribution of the pair plots is approximately oval in shape.
  # # Where the main axes of the ovals appears ia along a diagonal this indicates a correlation between features
  # plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

def pairwise_numericals(local_df, custom_colors):

  # Set the font size globally
  plt.rcParams.update({'font.size': 14})  # Change 14 to the desired font size

  # Linear relationships can be seen in most of the plots 
  # sns.set_style(style='white')

  # Define the order of species and corresponding colors
  hue_order = ['Adelie', 'Chinstrap', 'Gentoo']

  # Plot with specified hue order and palette - ss.pairplot ignores non-numerical features
  g = sns.pairplot(data=local_df, hue='species', hue_order=hue_order, palette=custom_colors) #, corner=True) #, height=6, aspect=8/6)
  g._legend.remove()

  # Change the axis labels
  axis_labels = ['bill length (mm)', 'bill depth (mm)','flipper length (mm)', 'body mass (g)']

  # change axis labels
  for ax, label in zip(g.axes[-1], axis_labels):
    ax.set_xlabel(label, fontsize=14)  # Set x-axis label

  for ax, label in zip(g.axes[:,0], axis_labels):
    ax.set_ylabel(label, fontsize=14)  # Set y-axis label

  custom_ticks = {
    0: [30,40,50,60],
    1: [14,16,18,20],  # Custom tick marks for the first subplot
    2: [180,200,220],          # Custom tick marks for the second subplot
    3: [3000,4000,5000,6000],          # Custom tick marks for the third subplot
  }

  # Override default tick marks for each subplot
  for i, ax in enumerate(g.axes.flatten()):
    if ax is not None and i in custom_ticks:
      ax.set_xticks(custom_ticks[i])
      ax.set_yticks(custom_ticks[i])

  # to change the legends location
  handles = g._legend_data.values()
  labels = g._legend_data.keys()
  g.figure.legend(handles=handles, labels=labels, title='species', loc='upper right', ncol=1)
  # g.figure.legend(handles=handles, labels=labels, loc='lower center', ncol=3)
  # g.figure.legend(handles=handles, labels=labels, loc='upper left', ncol=1)
  g.figure.subplots_adjust(top=0.92, bottom=0.08)
  plt.show()
 


  # # Set the font size globally
  # plt.rcParams.update({'font.size': 14})  # Change 14 to the desired font size

  # # Set the figure size and style
  # # plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
  # sns.set_style(style='white')

  # # Define the order of species and corresponding colors
  # hue_order = ['Adelie', 'Chinstrap', 'Gentoo']

  # # Plot with specified hue order and palette - ss.pairplot ignores non-numerical features
  # g = sns.pairplot(data=local_df, hue='species', hue_order=hue_order, palette=custom_colors, corner=True, height=6, aspect=8/6)

  # # Change the legend location
  # g._legend.set_bbox_to_anchor((1, 1))  # Adjust the legend location as needed

  # # Show the plot
  # plt.show()