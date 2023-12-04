# Import libraries
import matplotlib.pyplot as plt
import pandas as pd

# Make a larger than default plot
plt.figure(figsize=(12, 12))

# Read the data from the output file
file = pd.read_csv("output.csv")

# Create the 3D plot
plot = plt.axes(projection='3d')
plot.scatter3D(file.x, file.y, file.z, s=0.01, c=file.c, cmap='Paired')

# Title and labels for x, y, and z axes
plot.set_title("3D Scatter Plot")
plot.set_xlabel('danceability')
plot.set_ylabel('energy')
plot.set_zlabel('speechiness')

# Show the plot
plt.show()
