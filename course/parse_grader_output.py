

xIn = """[[ 1. 39. 87. 11. 43. 18. 20. 32. 14. 18. 42.]
 [ 1. 85. 71. 24. 18. 89. 31. 10. 64. 74. 41.]
 [ 1. 68. 76. 75. 62.  3. 12.  1. 69. 77.  5.]
 [ 1. 97. 10. 27. 25. 82. 11. 58.  6. 77. 23.]
 [ 1. 57. 30. 92. 61.  9.  2. 14. 15. 63. 85.]
 [ 1. 13. 38. 57.  4. 65. 55.  9.  9. 91. 59.]
 [ 1. 17. 80. 93.  1. 47.  7. 53.  6. 74. 71.]
 [ 1. 76. 14.  5. 36.  6. 73. 34. 62. 49. 15.]
 [ 1. 12. 75. 26. 53. 26. 79. 24. 77. 65.  1.]
 [ 1. 36. 53. 35. 18. 34. 64. 70. 86. 15. 45.]]"""

import re
from ast import literal_eval

import numpy as np


# put in a comma anywhere there is a space
# except next to start or end of array
are = re.sub(r"([^[])\s+([^]])", r"\1, \2", xIn)
a = np.array(literal_eval(are))


# selecting by condition in a 2d array
# mask = (z[:, 0] == 6)
# z[mask, :]
