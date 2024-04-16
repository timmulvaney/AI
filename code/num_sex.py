from globals import * 

# needed for dictionary
import ast 

def num_sex(local_df):

  # Create a figure and axes
  fig, axs = plt.subplots(2, 2, figsize=(12, 10))
  
  sns.set(font_scale=1)

  # Plot the first boxplot on the first subplot
  f = sns.boxplot(x=local_df.species, y=local_df.body_mass_g, hue=local_df.sex, ax=axs[0, 0])
  axs[0, 0].set_title(' ', fontsize = 14)
  axs[0, 0].set_xlabel(" ", fontsize = 14, x=0.5, labelpad=0)
  axs[0, 0].set_ylabel("body mass (g)", fontsize = 15)
  axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), size = 15)
  axs[0, 0].set_yticks(axs[0, 0].get_yticks())
  axs[0, 0].set_yticklabels(axs[0, 0].get_yticklabels(), size = 12)

  # Plot the second boxplot on the second subplot
  sns.boxplot(x=local_df.species, y=local_df.bill_length_mm, hue=local_df.sex, ax=axs[0, 1])
  axs[0, 1].set_title(' ', fontsize = 14)
  axs[0, 1].set_xlabel(" ", fontsize = 14, x=0.5, labelpad=0)
  axs[0, 1].set_ylabel("bill length (mm)", fontsize = 15)
  axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), size = 15)
  axs[0, 1].set_yticks(axs[0, 1].get_yticks())
  axs[0, 1].set_yticklabels(axs[0, 1].get_yticklabels(), size = 12)

  # Plot the third boxplot on the third subplot
  sns.boxplot(x=local_df.species, y=local_df.bill_depth_mm, hue=local_df.sex, ax=axs[1, 0])
  axs[1, 0].set_title(' ', fontsize = 14)
  axs[1, 0].set_xlabel(" ", fontsize = 14, x=0.5, labelpad=0)
  axs[1, 0].set_ylabel("bill depth (mm)", fontsize = 15)
  axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), size = 15)
  axs[1, 0].set_yticks(axs[1, 0].get_yticks())
  axs[1, 0].set_yticklabels(axs[1, 0].get_yticklabels(), size = 12)

  # Plot the fourth boxplot on the fourth subplot
  sns.boxplot(x=local_df.species, y=local_df.flipper_length_mm, hue=local_df.sex, ax=axs[1, 1])
  axs[1, 1].set_title(' ', fontsize = 14)
  axs[1, 1].set_xlabel(" ", fontsize = 14, x=0.5, labelpad=0)
  axs[1, 1].set_ylabel("flipper length (mm)", fontsize = 15)
  axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), size = 15)
  axs[1, 1].set_yticks(axs[1, 1].get_yticks())
  axs[1, 1].set_yticklabels(axs[1, 1].get_yticklabels(), size = 12)

  # Add title for the entire figure
  fig.suptitle('Physical features of the penguins by species and sex', fontsize=16)

  plt.tight_layout()  # Adjust layout
  plt.show()

  