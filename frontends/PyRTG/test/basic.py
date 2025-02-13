# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s --check-prefix=MLIR
# RUN: %rtgtool% %s --seed=0 --output-format=elaborated | FileCheck %s --check-prefix=ELABORATED
# RUN: %rtgtool% %s --seed=0 -o %t --output-format=asm && FileCheck %s --input-file=%t --check-prefix=ASM

from pyrtg import test, sequence, target, entry, rtg, Label, Set, Integer, Bag

# MLIR-LABEL: rtg.target @Tgt0 : !rtg.dict<entry0: !rtg.set<index>>
# MLIR-NEXT: [[C0:%.+]] = index.constant 0
# MLIR-NEXT: [[C1:%.+]] = index.constant 1
# MLIR-NEXT: [[SET:%.+]] = rtg.set_create [[C0:%.+]], [[C1:%.+]] : index
# MLIR-NEXT: rtg.yield [[SET]] : !rtg.set<index>
# MLIR-NEXT: }


@target
class Tgt0:

  @entry
  def entry0():
    return Set.create(Integer(0), Integer(1))


# MLIR-LABEL: rtg.target @Tgt1 : !rtg.dict<entry0: index, entry1: !rtg.label>
# MLIR-NEXT: [[C0:%.+]] = index.constant 0
# MLIR-NEXT: [[LBL:%.+]] = rtg.label_decl "l0"
# MLIR-NEXT: rtg.yield [[C0]], [[LBL]] : index, !rtg.label
# MLIR-NEXT: }


@target
class Tgt1:

  @entry
  def entry0():
    return Integer(0)

  @entry
  def entry1():
    return Label.declare("l0")


# MLIR-LABEL: rtg.sequence @seq0
# MLIR-SAME: ([[SET:%.+]]: !rtg.set<!rtg.label>)
# MLIR-NEXT: [[LABEL:%.+]] = rtg.set_select_random [[SET]]
# MLIR-NEXT: rtg.label local [[LABEL]]
# MLIR-NEXT: }


@sequence(Set.type(Label.type()))
def seq0(set: Set):
  set.get_random().place()


# MLIR-LABEL: rtg.sequence @seq1
# MLIR-NEXT: [[LABEL:%.+]] = rtg.label_decl "s1"
# MLIR-NEXT: rtg.label local [[LABEL]]
# MLIR-NEXT: }


@sequence()
def seq1():
  Label.declare("s1").place()


# MLIR-LABEL: rtg.test @test0
# MLIR-NEXT: }

# ELABORATED-LABEL: rtg.test @test0
# ELABORATED-NEXT: }

# ASM-LABEL: Begin of test0
# ASM: End of test0


@test()
def test0():
  pass


# MLIR-LABEL: rtg.test @test_args
# MLIR-SAME: (entry0 = [[SET:%.+]]: !rtg.set<index>)
# MLIR-NEXT: [[RAND:%.+]] = rtg.set_select_random [[SET]] : !rtg.set<index>
# MLIR-NEXT: rtg.label_decl "L_{{[{][{]0[}][}]}}", [[RAND]]
# MLIR-NEXT: rtg.label local
# MLIR-NEXT: }

# ELABORATED-LABEL: rtg.test @test_args_Tgt0
# CHECK: rtg.label_decl "L_0"
# CHECK-NEXT: rtg.label local
# CHECK-NEXT: }

# ASM-LABEL: Begin of test_args
# ASM-EMPTY:
# ASM-NEXT: L_0:
# ASM-EMPTY:
# ASM: End of test_args


@test(("entry0", Set.type(Integer.type())))
def test_args(set: Set):
  i = set.get_random()
  Label.declare(r"L_{{0}}", i).place()


# MLIR-LABEL: rtg.test @test_labels
# MLIR-NEXT: index.constant 5
# MLIR-NEXT: index.constant 3
# MLIR-NEXT: index.constant 2
# MLIR-NEXT: index.constant 1
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

# MLIR-NEXT: [[BAG0:%.+]] = rtg.bag_create (%idx2 x [[L0:%.+]], %idx1 x [[L1:%.+]]) : !rtg.label
# MLIR-NEXT: [[BAG1:%.+]] = rtg.bag_create (%idx1 x [[L2:%.+]]) : !rtg.label
# MLIR-NEXT: [[EMPTY_BAG:%.+]] = rtg.bag_create  : !rtg.label
# MLIR-NEXT: [[BAG2_1:%.+]] = rtg.bag_union [[BAG0]], [[BAG1]] : !rtg.bag<!rtg.label>
# MLIR-NEXT: [[BAG2:%.+]] = rtg.bag_union [[BAG2_1]], [[EMPTY_BAG]] : !rtg.bag<!rtg.label>
# MLIR-NEXT: [[RL2:%.+]] = rtg.bag_select_random [[BAG2]] : !rtg.bag<!rtg.label>
# MLIR-NEXT: [[SUB:%.+]] = rtg.bag_create (%idx1 x [[RL2]]) : !rtg.label
# MLIR-NEXT: [[BAG3:%.+]] = rtg.bag_difference [[BAG2]], [[SUB]] inf : !rtg.bag<!rtg.label>
# MLIR-NEXT: rtg.label local [[RL2]]
# MLIR-NEXT: [[BAG4:%.+]] = rtg.bag_difference [[BAG3]], [[BAG1]] : !rtg.bag<!rtg.label>
# MLIR-NEXT: [[RL3:%.+]] = rtg.bag_select_random [[BAG4]] : !rtg.bag<!rtg.label>
# MLIR-NEXT: rtg.label local [[RL3]]

# MLIR-NEXT: [[SEQ:%.+]] = rtg.get_sequence @seq0 : !rtg.sequence<!rtg.set<!rtg.label>>
# MLIR-NEXT: [[SUBST:%.+]] = rtg.substitute_sequence [[SEQ]]([[SET0]]) : !rtg.sequence<!rtg.set<!rtg.label>>
# MLIR-NEXT: [[RAND1:%.+]] = rtg.randomize_sequence [[SUBST]]
# MLIR-NEXT: rtg.embed_sequence [[RAND1]]
# MLIR-NEXT: [[RAND2:%.+]] = rtg.randomize_sequence [[SUBST]]
# MLIR-NEXT: rtg.embed_sequence [[RAND2]]
# MLIR-NEXT: [[RAND3:%.+]] = rtg.randomize_sequence [[SUBST]]
# MLIR-NEXT: rtg.embed_sequence [[RAND3]]

# MLIR-NEXT: [[SEQ1:%.+]] = rtg.get_sequence @seq1 : !rtg.sequence
# MLIR-NEXT: [[RAND4:%.+]] = rtg.randomize_sequence [[SEQ1]]
# MLIR-NEXT: rtg.embed_sequence [[RAND4]]

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

# ELABORATED-NEXT: rtg.label local
# ELABORATED-NEXT: rtg.label local

# ELABORATED-NEXT: [[L3:%.+]] = rtg.label_decl "l1_2"
# ELABORATED-NEXT: rtg.label local [[L3]]
# ELABORATED-NEXT: [[L4:%.+]] = rtg.label_decl "l1_3"
# ELABORATED-NEXT: rtg.label local [[L4]]
# ELABORATED-NEXT: rtg.label local [[L0]]

# ELABORATED-NEXT: [[L5:%.+]] = rtg.label_decl "s1"
# ELABORATED-NEXT: rtg.label local [[L5]]

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

# ASM-NEXT: l1_1:
# ASM-NEXT: l0:

# ASM-NEXT: l1_2:
# ASM-NEXT: l1_3:
# ASM-NEXT: l0:

# ASM-NEXT: s1:

# ASM-EMPTY:
# ASM: End of test_labels


@test()
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

  bag0 = Bag.create((2, l0), (1, l1))
  bag1 = Bag.create((1, l2))
  empty_bag = Bag.create_empty(rtg.LabelType.get())
  bag2 = bag0 + bag1 + empty_bag
  rl2 = bag2.get_random_and_exclude()
  rl2.place()

  bag2 -= bag1
  rl3 = bag2.get_random()
  rl3.place()

  seq0(set0)
  seq0.get()(set0)
  seq0.randomize(set0)()

  seq1()
