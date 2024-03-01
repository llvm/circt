from z3 import *
import numpy as np

s = Solver ()

array = [12, 45, 66, 34]

A = Array ('A', IntSort(), IntSort())
B = Array ('B', IntSort(), IntSort())
cnt = Array ('cnt', IntSort(), IntSort())

i = 0
for elem in array:
  A = Store(A, i, elem)
  i = i + 1

x = Int ('x')
s.add(x >= 0)
s.add(x < len(array))
s.add(Select(A, x) == 66)

if s.check() == sat:
  print s.model()
else:
  print "Not found!"