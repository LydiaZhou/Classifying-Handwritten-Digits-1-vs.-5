# Gradient Descent on a “Simple” Function

Consider the function f(x, y) = x^2 + 2y^2 + 2 sin(2πx) sin(2πy).

(a) Implement gradient descent to minimize this function. Let the initial values be x0 = 0.1; y0 =
0.1, let the learning rate be η = 0.01 and let the number of iterations be 50; Give a plot of
the how the function value drops with the number of iterations performed.
Repeat this problem for a learning rate of η = 0.1. What happened?

(b) Obtain the “minimum” value and the location of the minimum you get for gradient descent
using the same η and number of iterations as in part (a), starting from the following initial
points: (0.1, 0.1),(1, 1),(−0.5, −0.5),(−1, −1). A table with the location of the minimum
and the minimum values will suffice. You should now appreciate why finding the “true”
global minimum of an arbitrary function is a hard problem.
