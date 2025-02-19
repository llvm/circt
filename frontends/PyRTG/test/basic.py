# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s --check-prefix=MLIR
# RUN: %rtgtool% %s --seed=0 --output-format=elaborated | FileCheck %s --check-prefix=ELABORATED
# RUN: %rtgtool% %s --seed=0 -o %t --output-format=asm && FileCheck %s --input-file=%t --check-prefix=ASM

from pyrtg import test, rtg, Label, Set, Integer

# MLIR-LABEL: rtg.test @test0
# MLIR-NEXT: }

# ELABORATED-LABEL: rtg.test @test0
# ELABORATED-NEXT: }

# ASM-LABEL: Begin of test0
# ASM: End of test0


@test
def test0():
  pass


# MLIR-LABEL: rtg.test @test_labels
# MLIR-NEXT: index.constant 5
# MLIR-NEXT: index.constant 3
# MLIR-NEXT: [[L0:%.+]] = rtg.label_decl "l0"
# MLIR-NEXT: [[L1:%.+]] = rtg.label_unique_decl "l1"
# MLIR-NEXT: [[L2:%.+]] = rtg.label_unique_decl "l1"
# MLIR-NEXT: rtg.label global [[L0]]
# MLIR-NEXT: rtg.label external [[L1]]
# MLIR-NEXT: rtg.label local [[L2]]

# MLIR-NEXT: [[SET0:%.+]] = rtg.set_create [[L0]], [[L1]] : !rtg.label
# MLIR-NEXT: [[SET1:%.+]] = rtg.set_create [[L2]] : !rtg.label
# MLIR-NEXT: [[EMPTY_SET:%.+]] = rtg.set_create  : !rtg.label
# MLIR-NEXT: [[SET2_1:%.+]] = rtg.set_union [[SET0]], [[SET1]] : !rtg.set<!rtg.label>
# MLIR-NEXT: [[SET2:%.+]] = rtg.set_union [[SET2_1]], [[EMPTY_SET]] : !rtg.set<!rtg.label>
# MLIR-NEXT: [[RL0:%.+]] = rtg.set_select_random [[SET2]] : !rtg.set<!rtg.label>
# MLIR-NEXT: rtg.label local [[RL0]]
# MLIR-NEXT: [[SET2_MINUS_SET0:%.+]] = rtg.set_difference [[SET2]], [[SET0]] : !rtg.set<!rtg.label>
# MLIR-NEXT: [[RL1:%.+]] = rtg.set_select_random [[SET2_MINUS_SET0]] : !rtg.set<!rtg.label>
# MLIR-NEXT: rtg.label local [[RL1]]

# MLIR-NEXT: rtg.label_decl "L_{{[{][{]0[}][}]}}", %idx5
# MLIR-NEXT: rtg.label local
# MLIR-NEXT: rtg.label_decl "L_{{[{][{]0[}][}]}}", %idx3
# MLIR-NEXT: rtg.label local

# MLIR-NEXT: }

# ELABORATED-LABEL: rtg.test @test_labels
# ELABORATED-NEXT: [[L0:%.+]] = rtg.label_decl "l0"
# ELABORATED-NEXT: rtg.label global [[L0]]
# ELABORATED-NEXT: [[L1:%.+]] = rtg.label_decl "l1_0"
# ELABORATED-NEXT: rtg.label external [[L1]]
# ELABORATED-NEXT: [[L2:%.+]] = rtg.label_decl "l1_1"
# ELABORATED-NEXT: rtg.label local [[L2]]

# ELABORATED-NEXT: rtg.label local [[L0]]
# ELABORATED-NEXT: rtg.label local [[L2]]

# ELABORATED-NEXT: rtg.label_decl "L_5"
# ELABORATED-NEXT: rtg.label local
# ELABORATED-NEXT: rtg.label_decl "L_3"
# ELABORATED-NEXT: rtg.label local

# ELABORATED-NEXT: }

# ASM-LABEL: Begin of test_labels
# ASM-EMPTY:
# ASM-NEXT: .global l0
# ASM-NEXT: l0:
# ASM-NEXT: .extern l1_0
# ASM-NEXT: l1_1:

# ASM-NEXT: l0:
# ASM-NEXT: l1_1:

# ASM-NEXT: L_5:
# ASM-NEXT: L_3:

# ASM-EMPTY:
# ASM: End of test_labels


@test
def test_labels():
  l0 = Label.declare("l0")
  l1 = Label.declare_unique("l1")
  l2 = Label.declare_unique("l1")
  l0.place(rtg.LabelVisibility.GLOBAL)
  l1.place(rtg.LabelVisibility.EXTERNAL)
  l2.place()

  set0 = Set.create(l0, l1)
  set1 = Set.create(l2)
  empty_set = Set.create_empty(rtg.LabelType.get())
  set2 = set0 + set1 + empty_set
  rl0 = set2.get_random()
  rl0.place()

  set2 -= set0
  rl1 = set2.get_random_and_exclude()
  rl1.place()

  sub = Integer(1) - Integer(2)
  add = (sub & Integer(4) | Integer(3) ^ Integer(5))
  add += sub
  l3 = Label.declare(r"L_{{0}}", add)
  l3.place()
  l4 = Label.declare(r"L_{{0}}", 3)
  l4.place()
