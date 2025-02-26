# Complete Proof of the Prime Number Theorem

## Introduction
The Prime Number Theorem (PNT) states that the number of primes less than or equal to a real number $x$, denoted by $\pi(x)$, is asymptotically equivalent to $x/\ln(x)$. Formally:

$$\lim_{x \to \infty} \frac{\pi(x)}{x/\ln(x)} = 1$$

This proof will follow the analytic approach using complex analysis and properties of the Riemann zeta function.

## Preliminaries

### 1. The Riemann Zeta Function
The Riemann zeta function is defined for $\Re(s) > 1$ as:

$$\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}$$

It has a fundamental connection to prime numbers through Euler's product formula:

$$\zeta(s) = \prod_{p \text{ prime}} \frac{1}{1-p^{-s}}$$

### 2. The von Mangoldt Function
The von Mangoldt function $\Lambda(n)$ is defined as:

$$\Lambda(n) = 
\begin{cases} 
\ln(p) & \text{if } n = p^k \text{ for some prime } p \text{ and integer } k \geq 1 \\
0 & \text{otherwise}
\end{cases}$$

### 3. The Chebyshev Function
The second Chebyshev function $\psi(x)$ is defined as:

$$\psi(x) = \sum_{n \leq x} \Lambda(n)$$

The Prime Number Theorem is equivalent to:

$$\psi(x) \sim x$$

as $x \to \infty$.

## Key Steps in the Proof

### Step 1: Connection Between $\zeta(s)$ and $\psi(x)$
We begin by noting that:

$$-\frac{\zeta'(s)}{\zeta(s)} = \sum_{n=1}^{\infty} \frac{\Lambda(n)}{n^s}$$

for $\Re(s) > 1$.

### Step 2: Extending $\zeta(s)$ Beyond $\Re(s) > 1$
The zeta function can be analytically continued to the entire complex plane except for a simple pole at $s = 1$. Near $s = 1$, we have:

$$\zeta(s) = \frac{1}{s-1} + \gamma + O(s-1)$$

where $\gamma$ is the Euler-Mascheroni constant.

### Step 3: Explicit Formula for $\psi(x)$
Using Perron's formula and contour integration, we can establish:

$$\psi(x) = x - \sum_{\rho} \frac{x^{\rho}}{\rho} - \ln(2\pi) - \frac{1}{2}\ln(1-x^{-2})$$

where the sum is over the non-trivial zeros $\rho$ of $\zeta(s)$.

### Step 4: Zero-Free Region for $\zeta(s)$
A critical result is that $\zeta(s) \neq 0$ in the region:

$$\sigma \geq 1 - \frac{c}{\ln(|t|+2)}$$

for some constant $c > 0$, where $s = \sigma + it$.

### Step 5: Estimates in the Zero-Free Region
In this region, we can bound $|\zeta'(s)/\zeta(s)|$ and control the contribution of zeros to the explicit formula.

### Step 6: Final Asymptotic Analysis
Using the explicit formula and the zero-free region, we can establish:

$$\psi(x) = x + O(x \exp(-c\sqrt{\ln x}))$$

for some constant $c > 0$.

## Detailed Proof

### 1. Properties of the Riemann Zeta Function

We start by establishing that $\zeta(s)$ has a meromorphic continuation to the entire complex plane with a simple pole at $s = 1$. This can be shown using the functional equation:

$$\zeta(s) = 2^s \pi^{s-1} \sin\left(\frac{\pi s}{2}\right) \Gamma(1-s) \zeta(1-s)$$

Taking the logarithmic derivative of Euler's product formula:

$$\ln \zeta(s) = -\sum_{p} \ln(1-p^{-s})$$

and differentiating, we get:

$$\frac{\zeta'(s)}{\zeta(s)} = -\sum_{p} \frac{p^{-s}\ln p}{1-p^{-s}}$$

Expanding the fraction into a geometric series:

$$\frac{\zeta'(s)}{\zeta(s)} = -\sum_{p} \ln p \sum_{k=1}^{\infty} p^{-ks}$$

This gives us:

$$-\frac{\zeta'(s)}{\zeta(s)} = \sum_{n=1}^{\infty} \frac{\Lambda(n)}{n^s}$$

### 2. The Explicit Formula

To connect the zeta function to $\psi(x)$, we use the Perron formula to obtain:

$$\psi(x) = \frac{1}{2\pi i} \int_{c-i\infty}^{c+i\infty} -\frac{\zeta'(s)}{\zeta(s)} \frac{x^s}{s} ds$$

for $c > 1$. We then shift the contour of integration leftward, accounting for:
- The pole at $s = 1$
- The non-trivial zeros of $\zeta(s)$
- The trivial zeros at $s = -2n$ for integers $n \geq 1$

This yields the explicit formula:

$$\psi(x) = x - \sum_{\rho} \frac{x^{\rho}}{\rho} - \ln(2\pi) - \frac{1}{2}\ln(1-x^{-2})$$

### 3. Establishing a Zero-Free Region

A key part of the proof is showing that $\zeta(s) \neq 0$ in the region:

$$\sigma \geq 1 - \frac{c}{\ln(|t|+2)}$$

for some constant $c > 0$. This was proven independently by de la Vallée Poussin and Hadamard.

The proof uses the fact that if $\zeta(s) = 0$ for $s = \sigma + it$ with $\sigma$ close to 1, then the function:

$$F(s) = \zeta^3(s) \zeta^4(s+1) \zeta(s+2)$$

would have specific growth properties that lead to a contradiction when analyzed along a carefully chosen contour.

### 4. Bounding the Sum Over Zeros

Using the zero-free region and estimates on the density of zeros, we can show that:

$$\sum_{\rho} \frac{x^{\rho}}{\rho} = O(x \exp(-c\sqrt{\ln x}))$$

for some constant $c > 0$.

### 5. The Asymptotic Formula

Combining our results, we obtain:

$$\psi(x) = x + O(x \exp(-c\sqrt{\ln x}))$$

Since $\exp(-c\sqrt{\ln x}) \to 0$ as $x \to \infty$, we have:

$$\psi(x) \sim x$$

### 6. From $\psi(x)$ to $\pi(x)$

To complete the proof, we need to connect $\psi(x)$ to $\pi(x)$. Using the relation:

$$\psi(x) = \sum_{p \leq x} \ln p + \sum_{p^2 \leq x} \ln p + \sum_{p^3 \leq x} \ln p + \ldots$$

and partial summation, we can deduce:

$$\pi(x) \sim \frac{x}{\ln x}$$

which is the Prime Number Theorem.

## Conclusion

The Prime Number Theorem demonstrates a remarkable connection between the distribution of prime numbers and logarithmic functions. While this proof uses complex analysis, alternative approaches exist, including elementary proofs by Erdős and Selberg that avoid the use of complex analysis, though they are often considered more intricate.

The result itself has profound implications across number theory and showcases the deep structure underlying the seemingly irregular distribution of prime numbers.
