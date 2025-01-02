[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rulercosta/notebooks/blob/main/Essential_MML.ipynb)

# **1. LINEAR ALGEBRA**

Linear algebra is a branch of mathematics that is widely used throughout science and engineering. Linear algebra is a form of continuous mathematics. Linear algebra is the branch of mathematics concerning linear equations such as: linear maps, and their representations in vector spaces and through matrices. Linear algebra is central to almost all areas of mathematics.

---

## **1.0. Motivation for learning Linear Algebra**

There are several prerequisites for better understanding of Machine Learning and (especially) Deep Learning algorithms. In order to grasp the nitty gritties of higher ML concepts, one needs to have a strong base of higher mathematics.

Linear Algebra is the mathematical foundation that solves the problem of representing data as well as computations in machine learning models.

Hence, the concepts of linear algebra are crucial for understanding the theory behind machine learning, especially for deep learning.

---

## **1.1. Mathematical Objects**

Mathematical objects are what we talk and write about when we do math. Numbers, functions, triangles, matrices, groups and more complicated things such as vector spaces and infinite series are all examples of mathematical objects. Math objects are abstract objects. They are not physical objects, but we think about them and talk about them as if they actually existed. Math objects have certain properties that other kinds of abstract objects may not have. In particular, unlike other kinds of abstract objects, math objects are inert:
* Math objects don't move or change over time.
* Math objects don't interact with other objects or with the real world.

In particular, we are concerned with the study of following mathematical objects in Linear Algebra amongst others:

1. Scalars
2. Vectors
3. Matrices 
4. Tensors

Furthermore, it should be noted that higher mathematics is nothing but layers upon layers of abstraction. Intuitively, it would be right to consider **Sets** as the foundation of Linear Algebra-
* a line is a set of points
* a plane is a set of lines

Similarly,

* a vector is a set of points
* a matrix is a set of vectors

Therefore, an understanding of Set Theory is essential to grasp Linear Algebra with ease.

---

### 1.1.1. Scalars

A scalar is just a single number, in contrast to most of the other objects studied in linear algebra, which are usually arrays of multiple numbers. When we introduce them, we specify what kind of number they are. For example, we might say

$\text{Let}\ \ s\ \in\ \mathbb{R}\ \ \text{be the slope of the line}.$

while defining a real-valued scalar, or

$\text{Let}\ \ n\ \in\ \mathbb{N}\ \ \text{be the number of units}.$

while defining a natural number scalar.

### 1.1.2. Vectors

A vector is an array of numbers. The numbers are arranged in order. We can identify each individual number by its index in that ordering.

$\mathbf{x} := \begin{pmatrix} x_{1} \\ x_{2} \\ \vdots \\ x_{n} \end{pmatrix}, x_{i,j} \in \mathbb{R}$

First element of $\mathbf{x}$ is $\mathbf{x}_{1}$, the second element is $\mathbf{x}_{2}$, and so on.

If $\ \mathbf{x}_{i} \ \in \ \mathbb{R} \ \ \forall \ \ i \ \in \ \{1,2,3,...,n\}$, then the vector $\mathbf{x} \ $, having $\ n \ $ dimensions, lies in the set formed by taking the Cartesian product of $\ \mathbb{R} \ \ n$ times, denoted as $\ \mathbf{x} \ \in \ \mathbb{R}^{n}.$

One can think of vectors as identifying points in space, with each element giving the coordinate along a different axis. Sometimes we need to index a set of elements of a vector. In this case, we define a set containing the indices and write the set as a subscript.

For example, to access $\ \mathbf{x}_1$, $\ \mathbf{x}_3\ $ and $\ \mathbf{x}_6$, we define the set $\ S = \{1,3,6\}\ $ and write $\ \mathbf{x}_S.$ We use the $\ -\ $ sign to index the complement of a set. For example, $\ \mathbf{x}_{-1}\ $ is the vector containing all elements of $\ \mathbf{x}\ $ except for $\ \mathbf{x}_1.$ Similarly, $\ \mathbf{x}_S\ $ is the vector containing all elements of  $\ \mathbf{x}\ $ except for $\ \mathbf{x}_1$, $\ \mathbf{x}_3\ $ and $\ \mathbf{x}_6.$

$f(x) = y$
$\text{for $\ x \in \mathbf{x}$ and $\ y \in \mathbf{y}$}$

The ultimate goal of Machine Learning is learning functions from data, i.e., transformations or mappings from the domain onto the range of a function.

The domain $\ \mathbf{x} \ $ is usually a vector of variables or features mapping onto a vector of target values.

// ... content continues in similar fashion for remaining sections ...
