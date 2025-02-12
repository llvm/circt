# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s --check-prefix=MLIR
# RUN: %rtgtool% %s --seed=0 --output-format=elaborated | FileCheck %s --check-prefix=ELABORATED
# RUN: %rtgtool% %s --seed=0 -o %t --output-format=asm && FileCheck %s --input-file=%t --check-prefix=ASM

from pyrtg import test, Label, rtg

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
# MLIR-NEXT: [[L0:%.+]] = rtg.label_decl "l0"
# MLIR-NEXT: [[L1:%.+]] = rtg.label_unique_decl "l1"
# MLIR-NEXT: [[L2:%.+]] = rtg.label_unique_decl "l1"
# MLIR-NEXT: rtg.label global [[L0]]
# MLIR-NEXT: rtg.label external [[L1]]
# MLIR-NEXT: rtg.label local [[L2]]
# MLIR-NEXT: }

# ELABORATED-LABEL: rtg.test @test_labels
# ELABORATED-NEXT: [[L0:%.+]] = rtg.label_decl "l0"
# ELABORATED-NEXT: rtg.label global [[L0]]
# ELABORATED-NEXT: [[L1:%.+]] = rtg.label_decl "l1_0"
# ELABORATED-NEXT: rtg.label external [[L1]]
# ELABORATED-NEXT: [[L2:%.+]] = rtg.label_decl "l1_1"
# ELABORATED-NEXT: rtg.label local [[L2]]
# ELABORATED-NEXT: }

# ASM-LABEL: Begin of test_labels
# ASM-EMPTY:
# ASM-NEXT: .global l0
# ASM-NEXT: l0:
# ASM-NEXT: .extern l1_0
# ASM-NEXT: l1_1:
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
