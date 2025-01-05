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

### 1.2.3. Matrix multiplication:

If $\mathbf{A}$ is a matrix of size $m\times n$ and $\mathbf{B}$ is a matrix of size $n\times p$, then the product $\mathbf{AB}$ is a matrix of size $m\times p.$

### 1.2.4. Vectors:

A vector of length $n$ can be treated as a matrix of size $\ n \times 1,\ $ and the operations of vector addition, multiplication by scalars, and multiplying a matrix by a vector agree with the corresponding matrix operations.

### 1.2.5. Transpose:

If $\mathbf{A}$ is a matrix of size $m\times n$, then its transpose $\mathbf{A}^{\mathsf{T}}$ is a matrix of size $n\times m.$

### 1.2.6. Identity matrix:

An identity matrix is a matrix that does not change any vector when we multiply that vector by that matrix. We denote the identity matrix that preserves $n$-dimensional vectors as $\mathbf{I}_{n}.\ $ Formally, $\mathbf{I}_{n}\in\mathbb{R}_{n\times n},\ $ and

$\forall\ \mathbf{x}\in\mathbb{R}_n,\ \mathbf{I}_{n}\mathbf{x} = \mathbf{x}.$

$\mathbf{I}_{n}\ $ is the $n\times n$ identity matrix; its principle diagonal elements are equal to $1$ and its offdiagonal elements are equal to $0.$

### 1.2.7. Zero matrix:

It is denoted by $0$ the matrix of all zeroes (of relevant size).

### 1.2.8. Inverse:

If $\mathbf{A}$ is a square matrix, then its inverse $\mathbf{A}^{\mathsf{-1}}$ is a matrix of the same size. The matrices that have nonzero determinant have inverses, and are called invertible.

For square matrices,

$\mathbf{A}\mathbf{A}^{-1}=\mathbf{A}^{-1}\mathbf{A}=\mathsf{I}_{n}$

*In many cases, we can treat addition and multiplication of matrices as addition and multiplication of numbers. However, there are some differences between operations with matrices and operations with numbers:*

• Properties such as associative, distributive and commutative are followed in scalar multiplication and matrix addition.

• Matrix multiplication does not commute.

• In general, $\mathbf{AB} \not= \mathbf{BA}$, even if $\mathbf{A}$ and $\mathbf{B}$ are both square matrices. If $\mathbf{AB} = \mathbf{BA}$, then we say that $\mathbf{A}$ and $\mathbf{B}$ commute.

• For a general matrix $\mathbf{A}$, we cannot say that $\mathbf{AB} = \mathbf{AC}$ yields $\mathbf{B} = \mathbf{C}.$ (However, if we know that $\mathbf{A}$ is invertible, then we can multiply both sides of the equation $\mathbf{AB} = \mathbf{AC}$ to the left by $\mathbf{A}^{\mathsf{-1}}$ and get $\mathbf{B} = \mathbf{C}.)$

• The equation $\mathbf{AB} = 0$ does not necessarily yield $\mathbf{A}=0\ \text{ or }\ \mathbf{B}=0.$ For example, take:

$\mathbf{A}= \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix},\ \mathbf{B}= \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}.$

---

### A note on the methods of solving a system of linear equations:

Apart from the usual (direct) methods of solving a system of linear equations, which includes, Elimination Method, Substitution Method, and Cross multiplication Method, there are various other methods of solving a system of linear equations:

* Matrix Method:
  * Crammer's Rule
  * Gaussian Elimination
  * Gauss-Jordan Method
  * Traingularization Method
  * Choleskey Method
  * Partition Method

* Iterative methods:
  * Jacobi Iterative Method
  * Gauss-Seidel Iterative Method
  * SOR Method

It might seem overwhelming at first but as other properties of and special types of matrices are introduced, these methods will become easier to grasp. However, it must be noted that when it comes to solving a system of linear equations, in our case, we require a fast and efficient algorithm, so it would be fine not having some if not most of these methods on your fingertips.

**Note:** *This section will be expanded in the future.*

---

## **1.3. System of Linear Equations**

Using the linear algebra notations and the now defined mathematical objects, we can write a system of linear equations as follows:

$\begin{align}
\mathbf{A}\mathbf{x}=\mathbf{b}
\end{align}$

where $\mathbf{A}\in\mathbb{R}^{m\times n}\ $ is a known matrix, $\mathbf{b}\in\mathbb{R}^{m}\ $ is a known vector, and $\mathbf{x}\in\mathbb{R}^{n}\ $ is a vector of unknown variables.

In general, an element of vector $\mathbf{b}:$

$\begin{align}
\mathbf{b}_{i} = \mathbf{A}_{i,:}\ \mathbf{x}
\end{align}$

Matrix-vector product notation provides a more compact representation for equations of this form.

In order to solve for the unknown vector in the equation,

$\begin{align}
\mathbf{A}\mathbf{x}=\mathbf{b}
\end{align}$

pre-multiply both sides of the equation by the inverse of matrix $\ \mathbf{A},$

$\begin{align}
\mathbf{A}^{-1}\mathbf{A}\mathbf{x}& =\mathbf{A}^{-1}\mathbf{b}\\
\mathbf{I}_{n}\mathbf{x}& =\mathbf{A}^{-1}\mathbf{b}\\
\mathbf{x}& =\mathbf{A}^{-1}\mathbf{b}.
\end{align}$

When $\mathbf{A}^{-1}$ exists, several different algorithms can find it in closed form. In theory, the same inverse matrix can then be used to solve the equation many times for different values of $\mathbf{b}.\ $ However, $\mathbf{A}^{-1}\ $ is primarily useful as a theoretical tool and should not actually be used in practice for most software applications. Because $\mathbf{A}^{-1}\ $ can be represented with only limited precision on a digital computer, algorithms that make use of the value of $\mathbf{b}$ can usually obtain more accurate estimates of $\mathbf{x}$.

For $\mathbf{A}^{-1}$ to exist, equation $\ \mathbf{A}\mathbf{x}=\mathbf{b}\ $ must have *exactly one solution for every value of $\ \mathbf{b},$* i.e. unique value of $\ \mathbf{x}\ $ for every value of $\ \mathbf{b}.$

It is also possible for the system of equations to have no solutions or infinitely many solutions for some values of $\ \mathbf{b}.$ It is not possible, however, to have more than one but less than infinitely many solutions for a particular $\ \mathbf{b};\ $ if both $x\ \text{and}\ y$ are solutions, then

$\begin{align}
z = \alpha{x} + (1 −\alpha){y}
\end{align}$

is also a solution for any real $\ \alpha.$

To analyze how many solutions the equation has, think of the columns of $\ \mathbf{A}\ $ as specifying different directions we can travel in from the origin (the point specified by the vector of all zeros), then determine how many ways there are of reaching $\ \mathbf{b}.\ $ In this view, each element of $\mathbf{x}$ specifies how far we should travel in each of these directions, with $\ \mathbf{x}_{i}\ $ specifying how far to move in the direction of column i:

$\begin{align}
\mathbf{A}\mathbf{x}=\sum _{i}\mathbf{x}_{i}\mathbf{A}_{:,i}.
\end{align}$

In general, this kind of operation is called a ***Linear Combination.*** The span of a set of vectors is the set of all points obtainable by linear combination of the original vectors. This particular span is known as the **column space** or the **range** of $\ \mathbf{A}.$

Solution of the equation $\ \mathbf{A}\mathbf{x}=\mathbf{b}\ $ exists if $\ \mathbf{b}\ $ lies in the span of the columns of $\ \mathbf{A}.\ $ To have a solution for all values of $\ \mathbf{b}\in\mathbb{R}^{m},\ $ the column space of $\ \mathbf{A}\ $ must be all of $\ \mathbb{R}^{m}\ $ which implies that $\ \mathbf{A}\ $ must have at least $\ m\ $ columns $(n\geq m)$.

## **1.4. Norms**

In Machine Learning, the size of vectors is measured using a function called a **norm**. Formally, the $\ L^{p}\ $ norm is given by:

$\begin{align}
\|\mathbf{x}\|_{p} \  = \left( \sum_i \left| x_i \right| ^p \right) ^ {\frac {1}{p}}
\end{align}$

for $\ p \in \mathbb{R}, p \ge 1. \ $

The $\ L^{p}\ $ norm is a *mapping function.* It maps vectors to non-negative values and satisfies these properties:

* $\ f(x) = 0 \implies x = 0 \ $
* $\ f(x+y) \le f(x) + f(y)\ $ (the **triangle inequality**)
* $\ \forall \alpha \ \in \mathbb{R}, f(\alpha x) = \left| \alpha \right|f(x) \ $

For a vector $\ \mathbf{x} \ ,$ the most commonly used norm is the $\ L^{2} \ $ norm, also known as **Euclidean norm**:

$\begin{align}
\ \|\mathbf{x}\| = \sqrt{x_1^2 + \dots + x_n^2}
\end{align}$

The $\ L^{1} \ $ norm is used when it's necessary to differentiate between zero and non-zero elements:

$\begin{align}
\|\mathbf{x}\|_{1} = \sum_{i} \left|x_{i}\right|
\end{align}$

Two other important norms are:

* **Max norm**:
$\begin{align}
\|\mathbf{x}\|_{\infty} = \max_{i} \left|x_{i}\right|
\end{align}$

* **Frobenius norm** (for matrices):
$\begin{align}
\|A\|_{F} = \sqrt{\sum_{i,j}A^{2}_{i,j}}
\end{align}$

The dot product of two vectors can be written in terms of norms:

$\begin{align}
\mathbf{x}^{T}\mathbf{y} = \|\mathbf{x}\|_{2}\|\mathbf{y}\|_{2}\cos\theta
\end{align}$

where $\theta$ is the angle between $\mathbf{x}$ and $\mathbf{y}$.
