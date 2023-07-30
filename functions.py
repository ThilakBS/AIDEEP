# I need to import all the modules for my calculator
from builtins import function

import numpy as np
import matplotlib.pyplot as plt
import math
# need to define my variables for my plot points
x = np.linspace(-5, 5, 1000)


plt.plot(x, y, c='red')
plt.plot(x, x+5, c='blue')

plt.ylim(-5, 5)

plt.show()