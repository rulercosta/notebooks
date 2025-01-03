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

### 1.1.3. Matrices

A matrix is a 2-D array of numbers, so each element is identified by two indices instead of just one.

$\mathbf{A} := \begin{bmatrix} {A}_{1,1} & {A}_{1,2} & \ldots & {A}_{1,n}\\ {A}_{2,1} & {A}_{2,2} & \ldots & {A}_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ {A}_{m,1} & {A}_{m,2} & \ldots & {A}_{m,n}\end{bmatrix}, A_{i,j} \in \mathbb{R}$

If a real-valued matrix $\ \mathbf{A}\ $ has a height of $\ m\ $ (rows) and a width of $\ n\ $ (columns), then we say that $\ \mathbf{A}\ \in \ \mathbb{R}^{mn}.$

$i^{th}\ $ row of $\ \mathbf{A}\ $ is denoted by $\ \mathbf{A}_{i,:}\ $ and $j^{th}\ $ row of $\ \mathbf{A}\ $ is denoted by $\ \mathbf{A}_{:,j}.\ $

Suppose by applying some function $\ f\ $ to $\ \mathbf{A}\ $ the resultant matrix is $\ \mathbf{B}\ $ then the $\ (i,j)^{th}\ $ element of $\ \mathbf{B}\ $ is given by $\ f(\mathbf{A})_{i,j}.\ $

### 1.1.4. Tensors

A tensor is an array with more than two axes. In the general case, an array of numbers arranged on a regular grid with a variable number of axes is known as a tensor. Suppose a tensor named $\ \mathsf{A}\ $ has three dimensions then the element of $\ \mathsf{A}\ $ at coordinates $\ (i, j, k)\ $ is denoted as $\ \mathsf{A}_{i,j,k}.$

### A note on the transpose of matrices:

The transpose of a matrix is the mirror image of the matrix across its principle diagonal line. The transpose of a matrix $\ \mathbf{A}\ $ is denoted by $\ \mathbf{A}^{\mathsf{T}}.$

$\mathbf{A} = \begin{bmatrix} {A}_{1,1} & {A}_{1,2} & \ldots & {A}_{1,n}\\ {A}_{2,1} & {A}_{2,2} & \ldots & {A}_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ {A}_{m,1} & {A}_{m,2} & \ldots & {A}_{m,n}\end{bmatrix}$

$\mathbf{A}^{\mathsf{T}} = \begin{bmatrix} {A}_{1,1} & {A}_{2,1} & \ldots & {A}_{m,1}\\ {A}_{1,2} & {A}_{2,2} & \ldots & {A}_{m,2} \\ \vdots & \vdots & \ddots & \vdots \\ {A}_{1,n} & {A}_{2,n} & \ldots & {A}_{m,n}\end{bmatrix}$

Vectors can be thought of as matrices that contain only one column. Thus, the transpose of a vector is therefore a matrix with only one row. This can be denoted as $\ \mathbf{x} = {\begin{pmatrix}x_{1},x_{2},x_{3}\end{pmatrix}}^{\mathsf{T}}.$

A scalar can be thought of as a matrix with only a single entry. From this, we can see that a scalar is its own transpose: $\ a = a^{\mathsf{T}}.$

---

## **1.2. Properties of matrix operations**

The operations are as follows:

### 1.2.1. Addition:

If $\mathbf{A}$ and $\mathbf{B}$ are matrices of the same size $m\times n$, then $\mathbf{A} + \mathbf{B}$,
their sum, is a matrix of size $m\times n.$

### 1.2.2. Multiplication by scalars:

If $\mathbf{A}$ is a matrix of size $m\times n$ and $\alpha$ is a scalar,
then $\alpha\mathbf{A}$ is a matrix of size $m\times n.$

// ... I can continue with the remaining sections if you'd like ...
