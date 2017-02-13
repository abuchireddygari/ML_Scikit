import numpy as np
import matplotlib.pyplot as plt

# generate some random data
normal_dist = np.random.normal(0, 0.01, 1000)

# init figure
fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(2, 2, 1)
ax.plot(normal_dist)
ax.set_title('Line plot', size=20)

ax = fig.add_subplot(2, 2, 2)
ax.plot(normal_dist, 'o')
ax.set_title('Scatter plot', size=20)

ax = fig.add_subplot(2, 2, 3)
ax.hist(normal_dist, bins=50, color='b')
ax.set_title('Histogram', size=20)
ax.set_xlabel('count', size=16)

# ax = fig.add_subplot(2, 2, 4)
ax.boxplot(normal_dist)
ax.set_title('Boxplot', size=20)
plt.draw()
plt.show()

# Exercises
# - change marker type to a square in the scatter plot
# - try other letters and symbols
# - change color of the histogram to green
# - see if you can change the color of the Scatter Plot to red
# - add text to the plot using the ax.text() method
#   (doc here: http://matplotlib.org/users/text_intro.html)
# - see if you can write "Boxes are cool" in the boxplot?
# - add vertical and horizontal lines using the plt.axhline() and plt.axvline()
#   (doc here: http://matplotlib.org/examples/pylab_examples/axhspan_demo.html)
