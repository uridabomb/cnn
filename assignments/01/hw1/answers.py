r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

the selection of any $\Delta > 0$ is arbitrary because this effects
only the scale of the weights and not the quality of the linear seperation
as we learn, the weights change to optimaly seperate the data.
if the optimal solution for some $\Delta > 0$ is a weight matrix $\mat{W}$
than the optimal solution we learn for a margin of $\alpha\Delta$ will be $\alpha\mat{M}$

"""

part3_q2 = r"""
**Your answer:**

1. form the images we can see every separator has learned
 which pixels are most important to tell this class apart from others and what their values should be.
 for instance we see in all figures an area that is usually part of the figure (middle bottom and a bit right) but still gives almost no information about the classification
 this is not necessarily how we tell numbers apart for instance there is a sample of a 5 with a small bottom part that is mistaken for 6 because usually 5s have a large round open bottom
 we can also tell numbers apart by length relations
2. 

"""

part3_q3 = r"""
**Your answer:**

1. good, we see a constant decrease in the loss without any zigzags that occur with a learning rate too high.
yet we do manage to learn and reach what seems to be a plato which indicates that we cannot learn a lot more
2. Slightly overfitted to the training set - as we see that the train loss is below the validation loss yet the accuracy on the train set is higher than the validation set

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
