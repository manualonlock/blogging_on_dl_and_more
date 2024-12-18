# Intro

Hello! My name is Egor. Long story short, I've spent 5 years in the field of software engineering to this date. I like digging into other areas of computer science,
as well as those which aren't quite relevant to the kind of work I do. I've had my hands on robotics, low-level OS programming, network programming, web programming 
and lots of other stuff, and lately I've been also exploring the field of AI and Deep Learning, trying to get a sense of how these things work. I'll be posting my discoveries and braindumps here,
mostly to somehow organize the knowledge I'll be obtaining but also with a small chance of helping others. 

Also, it would be remiss of mine if I didn't mention [the FastAI Course](https://course.fast.ai/). Thanks to Jeremy Howard and lots of other enthusiasts, this outstanding course
has already helped lots of people to get their heads around AI. This course isn't just about AI, but also cultivates a certain view on how one should learn. Do check this out if you
want to know about AI and Deep Learning but have no idea where to start!


## Neural Nets vs Normal Programs

A very good example that demonstrates how neural networks are different from normal programs would be the following 2 pictures

### Normal Program
![image](https://github.com/user-attachments/assets/77e7444a-72e8-4bae-87f0-60cd98e55001)


### A Program With Weight Assignment
![image](https://github.com/user-attachments/assets/dc6e26c0-73be-4f0e-8b08-473c5a837c0e)


Don't worry if those pictures don't yet make sense as they shouldn't! They will start to very soon!

In the case of normal program, you always have an input, a program that does loops, if statements, variable assignments etc, and output. The problem with normal programs
is that they are dumb. They are way too demanding in how you tell them to behave. And the problem with being too explicit is that there're problems such that humans themselves solve, but
they have no idea how! Can you think of a program that recognizes cats from dogs? Try explaining how you do it yourself and you probably won't come up with the idea, hence it can't be laid out in a program.

Machine learning, on the other hand, is the way to solve exactly this kind of tasks! And to achieve this, we come up with something that is a little bit different to what a normal program is.
At first, we remove a program (please refer to the pictures pictures above while reading this) and replace it with a model, and along with the input we add weights.
Now the output our software produces depends on the input... and the weights! Weights are just variables that somehow change, and changing weights, that are fed into the model, change the model's behavior!

Now let's zoom out and see what the idea of weights really comes from. To not be tiresome, this is something similar to neurons, found in our brains.
Neurons are very complex, they are made up of a whole lot of things such as axons, presynaptic thermals, dentrits, synapse and etc. But most importantly, neurons can produce signals, 
pass it to other neurons that produce a new signal based on the one received before. Together they make our brains work!

### Human brain's neuron
![image](https://github.com/user-attachments/assets/f9b3effa-5729-40cc-97c4-12252d041a52)

### Artificial Neural Network
![image](https://github.com/user-attachments/assets/96bb991f-2431-40f3-9e1c-1416bcf606b8)


So now we have weights. Given that weights are remotely similar to neurons (though are different things and there's no goal to replicate neurons exactly the way they are represented in human brains),
it must be intuitive that there must be something flexible enough to solve any task just by varying these weights. This thing is actually what we call **a Neural Network**!
It's just a mathematical function that behaves extremely different depending on what weights are being fed into it. In theory, the more weights you have the more scalable is the neural network.


But how do we know that neural netorks are actually supposed to work? Despite of the incredible efficiency and sort of magic in neural nets, under the hood they just do lot's of additions and multiplications.
How can we be sure the set of weights always exists such that solves our problem? In fact, there's a proof showing that with finite number of weights and a function that satisfies specific properties 
(like being non-linear, non-constant and continious), any mathematical function can be approximated to any level of accuracy. This proof is called "the Universal Approximation Theorem", and serves as a backbone to machine learning.
With this idea, to come up with the solution to a problem we just need to find the right number of weights with their values.

### How the weights are found?
Despite of having the beautiful ideas of weights and the Univeral Approximation Theorem, we still have yet one problem to solve. How do we come up with the weights?
We don't expect weights to be found easily, moreover we definitely don't want to be finding weights for each problem. For that reason, we would probably like to have some sort of mechanism for automatically finding weights. Perhaps, we can start with a random set of weights, give our model a problem to solve
and have it adjust the weights to come closer to the needed level of accuracy. Using this idea a model would probably be able to learn how to solve a problem despite of what kind of problem it's given, but just by learning from it's experience.
The procedure is generally called "weight assignment", and conveniently enough, the way to do this also exists!

This is called "Stochastic Gradient Descent". This is a type of gradient descent made specifically for machine learning, where the idea is to optimize a dataset (weights) in parts by taking random subsets of the dataset, rather than trying to optimize the whole thing at once.
Which makes perfect sense, because to this day there're models with trillions of parameters (weights), and computing them at once would be too expensive.

### Putting this all together
The only thing left is that we need to somehow measure our model's performance. For the "cat vs dogs" kind of task, this would be quite easy, the performance of a model would be how often it gets the image right.
So we can just label our images and see what the model we train predicts. If it predicts wrong, then we call it loss, and Stochastic Gradient Descent works towards reducing the amount of loss.

Also it must be obvious that the neural network we choose to train isn't just alone, but there can be different types of neural networks, or more precisely their mathematical forms.
The mathematical form of a neural network is called **architecure**. You might have heard about convolutional neural networks or recurrent networks, so those basically represent a specific architecture.

**NOTE** please be aware weights are nowadays called parameters, so on the following picture you'll see parameters written down inside the ellipse, but this is what I previously referred to as "weights".

Let's revise what neural networks look like given all of the above
![image](https://github.com/user-attachments/assets/c9af2bbe-af5b-434f-b360-2c187ac5f05e)

Now, not only our model (or particular architecture) takes an input and weights, but also the difference lies in it's training process, where the predictions, compared to labels, produce loss,
and stochastic gradient descent tries to minimize the loss.


