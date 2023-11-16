from z3 import *

N = 32

IA = Function('IA', BitVecSort(N), BoolSort())
IB = Function('IB', BitVecSort(N), BoolSort())

nodes = ["A", "B"]
invariants = [IA, IB]

def AToB(cnt):
    return cnt

def BToA(cnt):
    return BitVecVal(0, N)

def BToB(cnt):
    return cnt + 1

def transitionAToB(cnt):
    return True

def transitionBToA(cnt):
    return cnt == 5

def transitionBToB(cnt):
    return cnt != 5


transition_matrix = [[None, transitionAToB], [transitionBToA, transitionBToB]]
transformation_matrix = [[None, AToB], [BToB, BToA]]

solver = Solver()

x = BitVec("x", N)

for a in range(len(nodes)):
    for b in range(len(nodes)):
        if transition_matrix[a][b] is None:
            continue

        solver.add(ForAll([x], Implies(And(invariants[a](x), transition_matrix[a][b](transformation_matrix[a][b](x))), invariants[b](x))))
solver.add(IA(0))
# solver.add(ForAll([x], Implies(IA(x), IB(AToB(x)))))
# solver.add(ForAll([x], Implies(And(IB(x), transitionBToA(x)), IA(BToA(x)))))
# solver.add(ForAll([x], Implies(And(IB(x), transitionBToB(x)), IB(BToB(x)))))

# solver.add(ForAll([x], Implies(And(IA(x), x != 0), False)))
solver.add(ForAll([x], Implies(And(IB(x), x > 5), False)))
print(solver.check())
print(solver.model())
print(solver)
