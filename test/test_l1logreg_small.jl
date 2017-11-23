using ProximalOperators

A = [  1.0  -2.0   3.0  -4.0  5.0;
       2.0  -1.0   0.0  -1.0  3.0;
      -1.0   0.0   4.0  -3.0  2.0;
      -1.0  -1.0  -1.0   1.0  3.0]
b = [1.0, 2.0, 3.0, 4.0]

m, n = size(A)

f = Translate(LogisticLoss(), -b)
lam = 0.1
g = NormL1(lam)

x_star = [0, 0, 2.114635341704963e-01, 0, 2.845881348733116e+00]
