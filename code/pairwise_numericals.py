from globals import * 

def pairwise_numericals(df, custom_colors):



  # Multiple Bivariate distribution
  # Linear relationships can be seen in most of the plots 
  sns.set_style(style='white')

  # Define the order of species and corresponding colors
  hue_order = ['Adelie', 'Chinstrap', 'Gentoo']
  # palette = {'Adelie': '#01193f', 'Chinstrap': '#6baddf', 'Gentoo': '#d2b486'}

  # Plot with specified hue order and palette - ss.pairplot ignores non-numerical features
  sns.pairplot(data=df, hue='species', hue_order=hue_order, palette=custom_colors)

  # For each species, the joint distribution of the pair plots is approximately oval in shape.
  # Where the main axes of the ovals appears ia along a diagonal this indicates a correlation between features
  plt.show()
 