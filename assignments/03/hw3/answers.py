r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
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
Explain the effect of depth on the accuracy. What depth produces the best results and why do you think that's the case?
Were there values of L for which the network wasn't trainable? what causes this? Suggest two things which may be done to resolve it at least partially.

As the depth gets bigger, the accuracy get smaller. As we can see, the best results are for L=2,4. L=8 has a bit lower 
accuracy. For L=16 the network was not trainable. A reasonable explaination for those results is 
the vanishing/exploding gradient phenomena.

There are several things that may be done to resolve this problem. Among them:
1. Add batch normalizations, which will make the gradients to be more stable.
2. Use skip-connections, which will make the network to do less multiplications along the depth, and the gradient will 
get bigger/smaller in a lower rate.
"""

part3_q2 = r"""
**Your answer:**
As the best K vary for each L=2,4,8, we may analyze that there is no K that is better to use than the rest.
In comparison to experiment 1.1, increase in K does not decrease the accuracy, as opposed to increase in L
(from some threshold) in experminent. 

Moreover, in experiment 1.2, there is no value of K which yields an untrainable
network.
"""

part3_q3 = r"""
**Your answer:**
The best results are for L=2, as the network's depth is okay, while L=4 yielded untrainable network as the network is
too deep. 

Those results fit nicely with our prior experiments, as we saw there is no best K and the network should not 
be too deep or too shallow.
"""


part3_q4 = r"""
**Your answer:**
We added batch normalization and dropout layers, in order to make the network be more trainable for more epochs,
and to ba able to handle larger depths. Moreover, we replaced the linear layers with avarage pooling layer
with output size (1, 1).

L3_K64-128-256 has the best accuracy. L4_K64-128-256-512 has the worst result, 
as even with normalization this network is too deep. 

In comparison to experiment 1, we can see that the early-stopping mechainsm stopped the training after much more epochs, 
and the test accuracy correlates much better with the train accuracy (they increase in a much closer rate).

Moreover, the best final test accuracy improved significantly.
"""

# ==============
# Part 4 answers


def part4_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "KING RAN:"
    temperature = .1
    # ========================
    return start_seq, temperature


part4_q1 = r"""

for 2 main reasons:
- characters that are far from each other are less semantically connected or relevant to each other. 
splitting to sequences prevents the loss from being effected by far, hence less relevant chars.
- splitting to shorter sequences sequences speeds up the training - each batch takes less time to calaulate and propogate changes to the model.


"""

part4_q2 = r"""
- the method for the model to "remember" more than sequence length is with the "hidden state" of the GRU.


"""

part4_q3 = r"""
unlike other learning tasks, here the order of samples is important. as we mentioned, past sequences influence the hidden state.
hence -  it's important to feed them in correct order and not shuffle so that the "hidden state" represents the context accuratly..

"""

part4_q4 = r"""
**Your answer:**
- while sampling, we want out text to look structured and prevent mistakes and "random feeling" in the text.
this is why we lower the temperature and "pushing" the more likely outputs to being even more likely.

- very high temperature makes the chracter distribution very "soft": all characters have similar probability, and text looks more random.

- very low temperature makes the probability distribution more "steep" -the largest logit is pushed close to 1, and the sampling becomes very conservative and non-diverse. similar sequences are likely to be generated many times. 

"""
# ==============


# ==============
# Part 5 answers

PART5_CUSTOM_DATA_URL = None


def part5_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
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
# Part 6 answers

PART6_CUSTOM_DATA_URL = None


def part6_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""