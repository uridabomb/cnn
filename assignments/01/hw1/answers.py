r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**



Increasing `k` increases the generalisation until a cerain point, we can see in the graph that after `k = 3` the accuracy starts to decline.
this happens because we let more samples that might be farther apart and of the wrong class. to emphasize this point, let's think of what happens when `k == N` (`N` is size of dataset) in that case we just return the majority label in the dataset.

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

1. From the images we can see that every separator has learned
 which pixels are most important to tell this class apart from others and what their values should be.
 For instance, we see in all figures an area that is usually part of the figure (middle bottom and a bit right) but still gives almost no information about the classification
 this is not necessarily how we tell numbers apart. For instance, there is a sample of a 5 with a small bottom part that is mistaken for 6 because usually 5s have a large round open bottom.
 We can also tell numbers apart by length relations.
2. This is different from KNN as the KNN classifies by taking into consideration all the pixels equally.

"""

part3_q3 = r"""
**Your answer:**

1. Good, we see a constant decrease in the loss without any zigzags that occur with a learning rate too high.
yet we do manage to learn and reach what seems to be a plato which indicates that we cannot learn a lot more
2. Slightly overfitted to the training set - as we see that the train loss is below the validation loss yet the accuracy on the train set is higher than the validation set

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern is a constant line at y=0. We can say that our residual graph seems pretty similar to that, hence our model fitted relatively okay.
The CV graph looks more packed around 0 than the top-5 graph, so we can infer that after CV the fitting is better.

"""

part4_q2 = r"""
**Your answer:**

Using the logspace allows us to explore values accross different orders of magnitude without sampling huge amounts of lambdas.

The model is fitted `k_folds * len(degree_range) * len(lambda_range)` number of times (not including the final fit).

"""

# ==============
