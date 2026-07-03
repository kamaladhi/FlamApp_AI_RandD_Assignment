# Parametric Curve Fitting & Geometric Transformation

This repository contains the solution for the Parametric Curve Fitting assessment.

## 🎯 1. Final Extracted Variables

The precise values for the unknown variables are:

- **$\theta$ (Theta)** = $30^{\circ}$ (or $0.5236$ radians)
- **$X$** = $55$
- **$M$** = $0.03$

### Desmos Format

As required in the PDF, here is the final equation in LaTeX format that can be copied directly to the Desmos calculator:

```latex
\left(t*\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55, 42+t*\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```

---

## 📁 2. Repository Structure

📦 FlamApp_AI_RandD_Assignment
 ┣ 📂 Dataset
 ┃ ┗ 📜 xy_data.csv        # The original point-cloud dataset (t is unknown)

 ┣ 📂plots

 ┃ ┗ 🖼️ visual_proof.png      # Matplotlib output proving the curve fit
 ┣ 📜 README.md            # Mathematical proofs and exact final variables
 ┣ 📜 solve.py             # Analytical OOP solver and visualization script

 ┗📜 requirements.txt     # Python dependencies

---

## 🧠 3. Engineering Approach & Mathematical Intuition

### Initial problem and Dataset analysis

After analysing the problem statement and the dataset I found a critical constraint that the provided `xy_data.csv` file contains 1500 un-indexed coordinate pairs (x,y). It is completely missing the independent time variable $t$.

Without the independent time $t$ parameter, determining each point on the curve is very difficult in a mapping problem. If we use standard machine learning or library-based curve fitting algorithms, it will not be able to determine the exact value of independent time variable $t$ and also it will not be able to give the exact value of the parameters $\theta, M, X$.

This understanding made me clear that the problem was designed to be solved mathematically not algorithmically.

### The Problem with Brute-Force

A standard approach to curve fitting on un-indexed point clouds is to use an iterative Euclidean distance minimization (finding the closest point on the curve for every data point). However, because the independent variable $t$ is unknown for the data in `xy_data.csv`, a brute-force solver would have $O(N \cdot M)$ complexity and would be highly prone to falling into local minima due to the trigonometric components.

### The "Hidden" Transformation Matrix

By analysing the algebraic structure of the parametric equation , I found that they are not arbitrary.
They consist of a base curve which is simple and that has been structurally altered by a  **2D Rotation Matrix** and a **Translation Vector**.

If we define a base curve as:

- $x_{base} = t$
- $y_{base} = e^{M|t|} \cdot \sin(0.3t)$

The equations below is a base curve multiplied by a standard rotation matrix for angle $\theta$, along with the translation of $(X, 42)$:

$$
\begin{pmatrix} x \cr y \end{pmatrix} = \begin{pmatrix} \cos(\theta) & -\sin(\theta) \cr \sin(\theta) & \cos(\theta) \end{pmatrix} \begin{pmatrix} t \cr e^{M|t|} \sin(0.3t) \end{pmatrix} + \begin{pmatrix} X \cr 42 \end{pmatrix}
$$

### The O(N) Solution: Inverse Rotation

By Analysing this geometric structure, I understood that bruteforce method will be very inefficient and complicated to implement and also it will lead to leakage of local minima. So I applied an **Inverse Rotation Matrix** algebraically to decouple the equations, Which allows the explicit calculation of the time-step $t$ for any arbitrary point $(x_i, y_i)$:

$$
t_i = (x_i - X)\cos(\theta) + (y_i - 42)\sin(\theta)
$$

By Decoupling the equations, I converted the non-convex point-cloud optimization problem into a mathematically precise loss function, which will converge to the exact required parameters in a fraction of a second.

### Global Optimization

I integrated the mathematical model inside a `ParametricCurveFitter` class in Python. I have also used `scipy.optimize.differential_evolution` to navigate the bounded parameter space. Because the inverse rotation evaluates in $O(N)$ time complexity, the global optimizer converges to the exact required parameters in a fraction of a second, yielding a very low Mean Squared Error of **$1.21 \times 10^{-11}$**.

## 📈 4. Usage & Visual Proof

To verify that the mathematical model perfectly fits the raw data, the script includes a visualization component.

1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the solver:
   `python solve.py`
3. The script will output the exact parameters to the console and automatically generate a `visual_proof.png` image. This image plots the exact mathematical curve over the original point cloud, visually proving the correctness of the analytical rotation model.

---
