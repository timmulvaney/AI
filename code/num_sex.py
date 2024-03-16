from globals import * 

# needed for dictionary
import ast 

def num_sex(df):

  # Create a figure and axes
  fig, axs = plt.subplots(2, 2, figsize=(12, 10))

  # Plot the first boxplot on the first subplot
  sns.boxplot(x=df.species, y=df.body_mass_g, hue=df.sex, ax=axs[0, 0])
  axs[0, 0].set_title('Body Mass', weight='bold')
  axs[0, 0].set_xlabel("penguin species", weight='bold', x=0.5, labelpad=0)
  axs[0, 0].set_ylabel("body mass (g)", weight='bold')

  # Plot the second boxplot on the second subplot
  sns.boxplot(x=df.species, y=df.bill_length_mm, hue=df.sex, ax=axs[0, 1])
  axs[0, 1].set_title('Bill Length', weight='bold')
  axs[0, 1].set_xlabel("penguin species", weight='bold', x=0.5, labelpad=0)
  axs[0, 1].set_ylabel("bill length (mm)", weight='bold')

  # Plot the third boxplot on the third subplot
  sns.boxplot(x=df.species, y=df.bill_depth_mm, hue=df.sex, ax=axs[1, 0])
  axs[1, 0].set_title('Bill Depth', weight='bold')
  axs[1, 0].set_xlabel("penguin species", weight='bold', x=0.5, labelpad=0)
  axs[1, 0].set_ylabel("bill depth (mm)", weight='bold')

  # Plot the fourth boxplot on the fourth subplot
  sns.boxplot(x=df.species, y=df.flipper_length_mm, hue=df.sex, ax=axs[1, 1])
  axs[1, 1].set_title('Flipper Length', weight='bold')
  axs[1, 1].set_xlabel("penguin species", weight='bold', x=0.5, labelpad=0)
  axs[1, 1].set_ylabel("lipper length (mm)", weight='bold')

  # Add title for the entire figure
  fig.suptitle('Physical features of the penguins by species and sex', weight='bold', fontsize=16)

  plt.tight_layout()  # Adjust layout
  plt.show()

  