# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s --check-prefix=MLIR
# RUN: %rtgtool% %s --seed=0 --output-format=elaborated | FileCheck %s --check-prefix=ELABORATED
# RUN: %rtgtool% %s --seed=0 -o %t --output-format=asm && FileCheck %s --input-file=%t --check-prefix=ASM

from pyrtg import test, sequence, target, entry, rtg, Label, Set, Integer, Bag, rtgtest, Immediate, IntegerRegister, Array, Bool

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


# MLIR-LABEL: rtg.target @Tgt1 : !rtg.dict<entry0: index, entry1: !rtg.isa.label>
# MLIR-NEXT: [[C0:%.+]] = index.constant 0
# MLIR-NEXT: [[LBL:%.+]] = rtg.label_decl "l0"
# MLIR-NEXT: rtg.yield [[C0]], [[LBL]] : index, !rtg.isa.label
# MLIR-NEXT: }


@target
class Tgt1:

  @entry
  def entry0():
    return Integer(0)

  @entry
  def entry1():
    return Label.declare("l0")


# MLIR-LABEL: rtg.target @Tgt4
# MLIR-NEXT: [[IDX12:%.+]] = index.constant 12
# MLIR-NEXT: [[IDX11:%.+]] = index.constant 11
# MLIR-NEXT: [[IDX10:%.+]] = index.constant 10
# MLIR-NEXT: [[IDX0:%.+]] = index.constant 0
# MLIR-NEXT: [[IDX1:%.+]] = index.constant 1
# MLIR-NEXT: [[IDX2:%.+]] = index.constant 2
# MLIR-NEXT: [[IDX3:%.+]] = index.constant 3
# MLIR-NEXT: [[ARR0:%.+]] = rtg.array_create [[IDX1]], [[IDX2]], [[IDX3]] : index
# MLIR-NEXT: [[RES0:%.+]] = rtg.array_extract [[ARR0]][[[IDX2]]] : !rtg.array<index>
# MLIR-NEXT: [[ARR1:%.+]] = rtg.array_create : !rtg.array<index>
# MLIR-NEXT: [[ARR2:%.+]] = rtg.array_create [[IDX0]], [[IDX1]], [[IDX2]] : index
# MLIR-NEXT: [[ARR3:%.+]] = rtg.array_create [[IDX10]], [[IDX11]], [[IDX12]] : index
# MLIR-NEXT: [[ARR4:%.+]] = rtg.array_create [[ARR2]], [[ARR3]] : !rtg.array<index>
# MLIR-NEXT: rtg.yield [[RES0]], [[ARR1]], [[ARR4]] : index, !rtg.array<!rtg.array<index>>, !rtg.array<!rtg.array<index>>


@target
class Tgt4:

  @entry
  def arr0():
    arr0 = Array.create([Integer(1), Integer(2), Integer(3)], Integer.type())
    return arr0[2]

  @entry
  def arr1():
    return Array.create([], Array.type(Integer.type()))

  @entry
  def arr2():
    return Array.create([
        Array.create([Integer(y * 10 + x)
                      for x in range(3)], Integer.type())
        for y in range(2)
    ], Array.type(Integer.type()))


# MLIR-LABEL: rtg.sequence @seq0
# MLIR-SAME: ([[SET:%.+]]: !rtg.set<!rtg.isa.label>)
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


# MLIR-LABEL: rtg.test @test1_args
# MLIR-SAME: (entry0 = [[SET:%.+]]: !rtg.set<index>)
# MLIR-NEXT: [[RAND:%.+]] = rtg.set_select_random [[SET]] : !rtg.set<index>
# MLIR-NEXT: rtg.label_decl "L_{{[{][{]0[}][}]}}", [[RAND]]
# MLIR-NEXT: rtg.label local
# MLIR-NEXT: }

# ELABORATED-LABEL: rtg.test @test1_args_Tgt0
# CHECK: rtg.label_decl "L_0"
# CHECK-NEXT: rtg.label local
# CHECK-NEXT: }

# ASM-LABEL: Begin of test1_args
# ASM-EMPTY:
# ASM-NEXT: L_0:
# ASM-EMPTY:
# ASM: End of test1_args


@test(("entry0", Set.type(Integer.type())))
def test1_args(set: Set):
  i = set.get_random()
  Label.declare(r"L_{{0}}", i).place()


# MLIR-LABEL: rtg.test @test2_labels
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

# MLIR-NEXT: [[SET0:%.+]] = rtg.set_create [[L0]], [[L1]] : !rtg.isa.label
# MLIR-NEXT: [[SET1:%.+]] = rtg.set_create [[L2]] : !rtg.isa.label
# MLIR-NEXT: [[EMPTY_SET:%.+]] = rtg.set_create  : !rtg.isa.label
# MLIR-NEXT: [[SET2_1:%.+]] = rtg.set_union [[SET0]], [[SET1]] : !rtg.set<!rtg.isa.label>
# MLIR-NEXT: [[SET2:%.+]] = rtg.set_union [[SET2_1]], [[EMPTY_SET]] : !rtg.set<!rtg.isa.label>
# MLIR-NEXT: [[RL0:%.+]] = rtg.set_select_random [[SET2]] : !rtg.set<!rtg.isa.label>
# MLIR-NEXT: rtg.label local [[RL0]]
# MLIR-NEXT: [[SET2_MINUS_SET0:%.+]] = rtg.set_difference [[SET2]], [[SET0]] : !rtg.set<!rtg.isa.label>
# MLIR-NEXT: [[RL1:%.+]] = rtg.set_select_random [[SET2_MINUS_SET0]] : !rtg.set<!rtg.isa.label>
# MLIR-NEXT: rtg.label local [[RL1]]

# MLIR-NEXT: rtg.label_decl "L_{{[{][{]0[}][}]}}", %idx5
# MLIR-NEXT: rtg.label local
# MLIR-NEXT: rtg.label_decl "L_{{[{][{]0[}][}]}}", %idx3
# MLIR-NEXT: rtg.label local

# MLIR-NEXT: [[BAG0:%.+]] = rtg.bag_create (%idx2 x [[L0:%.+]], %idx1 x [[L1:%.+]]) : !rtg.isa.label
# MLIR-NEXT: [[BAG1:%.+]] = rtg.bag_create (%idx1 x [[L2:%.+]]) : !rtg.isa.label
# MLIR-NEXT: [[EMPTY_BAG:%.+]] = rtg.bag_create  : !rtg.isa.label
# MLIR-NEXT: [[BAG2_1:%.+]] = rtg.bag_union [[BAG0]], [[BAG1]] : !rtg.bag<!rtg.isa.label>
# MLIR-NEXT: [[BAG2:%.+]] = rtg.bag_union [[BAG2_1]], [[EMPTY_BAG]] : !rtg.bag<!rtg.isa.label>
# MLIR-NEXT: [[RL2:%.+]] = rtg.bag_select_random [[BAG2]] : !rtg.bag<!rtg.isa.label>
# MLIR-NEXT: [[SUB:%.+]] = rtg.bag_create (%idx1 x [[RL2]]) : !rtg.isa.label
# MLIR-NEXT: [[BAG3:%.+]] = rtg.bag_difference [[BAG2]], [[SUB]] inf : !rtg.bag<!rtg.isa.label>
# MLIR-NEXT: rtg.label local [[RL2]]
# MLIR-NEXT: [[BAG4:%.+]] = rtg.bag_difference [[BAG3]], [[BAG1]] : !rtg.bag<!rtg.isa.label>
# MLIR-NEXT: [[RL3:%.+]] = rtg.bag_select_random [[BAG4]] : !rtg.bag<!rtg.isa.label>
# MLIR-NEXT: rtg.label local [[RL3]]

# MLIR-NEXT: [[SEQ:%.+]] = rtg.get_sequence @seq0 : !rtg.sequence<!rtg.set<!rtg.isa.label>>
# MLIR-NEXT: [[SUBST:%.+]] = rtg.substitute_sequence [[SEQ]]([[SET0]]) : !rtg.sequence<!rtg.set<!rtg.isa.label>>
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

# ELABORATED-LABEL: rtg.test @test2_labels
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

# ASM-LABEL: Begin of test2_labels
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
# ASM: End of test2_labels


@test()
def test2_labels():
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


# MLIR-LABEL: rtg.test @test3_registers_and_immediates()
# MLIR-NEXT: [[IMM32:%.+]] = rtg.constant #rtg.isa.immediate<32, 32>
# MLIR-NEXT: [[IMM21:%.+]] = rtg.constant #rtg.isa.immediate<21, 16>
# MLIR-NEXT: [[IMM13:%.+]] = rtg.constant #rtg.isa.immediate<13, 9>
# MLIR-NEXT: [[T2:%.+]] = rtg.fixed_reg #rtgtest.t2 : !rtgtest.ireg
# MLIR-NEXT: [[IMM5:%.+]] = rtg.constant #rtg.isa.immediate<5, 4>
# MLIR-NEXT: [[T1:%.+]] = rtg.fixed_reg #rtgtest.t1 : !rtgtest.ireg
# MLIR-NEXT: [[IMM12:%.+]] = rtg.constant #rtg.isa.immediate<12, 8>
# MLIR-NEXT: [[T0:%.+]] = rtg.fixed_reg #rtgtest.t0 : !rtgtest.ireg
# MLIR-NEXT: [[VREG:%.+]] = rtg.virtual_reg [#rtgtest.t0 : !rtgtest.ireg, #rtgtest.t1 : !rtgtest.ireg, #rtgtest.t2 : !rtgtest.ireg, #rtgtest.t3 : !rtgtest.ireg, #rtgtest.t4 : !rtgtest.ireg, #rtgtest.t5 : !rtgtest.ireg, #rtgtest.t6 : !rtgtest.ireg, #rtgtest.a7 : !rtgtest.ireg, #rtgtest.a6 : !rtgtest.ireg, #rtgtest.a5 : !rtgtest.ireg, #rtgtest.a4 : !rtgtest.ireg, #rtgtest.a3 : !rtgtest.ireg, #rtgtest.a2 : !rtgtest.ireg, #rtgtest.a1 : !rtgtest.ireg, #rtgtest.a0 : !rtgtest.ireg, #rtgtest.s1 : !rtgtest.ireg, #rtgtest.s2 : !rtgtest.ireg, #rtgtest.s3 : !rtgtest.ireg, #rtgtest.s4 : !rtgtest.ireg, #rtgtest.s5 : !rtgtest.ireg, #rtgtest.s6 : !rtgtest.ireg, #rtgtest.s7 : !rtgtest.ireg, #rtgtest.s8 : !rtgtest.ireg, #rtgtest.s9 : !rtgtest.ireg, #rtgtest.s10 : !rtgtest.ireg, #rtgtest.s11 : !rtgtest.ireg, #rtgtest.s0 : !rtgtest.ireg, #rtgtest.ra : !rtgtest.ireg, #rtgtest.sp : !rtgtest.ireg]
# MLIR-NEXT: rtgtest.rv32i.addi [[VREG]], [[T0]], [[IMM12]]
# MLIR-NEXT: rtgtest.rv32i.slli [[VREG]], [[T1]], [[IMM5]]
# MLIR-NEXT: rtgtest.rv32i.beq [[VREG]], [[T2]], [[IMM13]] : !rtg.isa.immediate<13>
# MLIR-NEXT: rtgtest.rv32i.jal [[VREG]], [[IMM21]] : !rtg.isa.immediate<21>
# MLIR-NEXT: rtgtest.rv32i.auipc [[VREG]], [[IMM32]] : !rtg.isa.immediate<32>
# MLIR-NEXT: }


@test()
def test3_registers_and_immediates():
  vreg = IntegerRegister.virtual()
  imm12 = Immediate(12, 8)
  rtgtest.ADDI(vreg, IntegerRegister.t0(), imm12)
  rtgtest.SLLI(vreg, IntegerRegister.t1(), Immediate(5, 4))
  rtgtest.BEQ(vreg, IntegerRegister.t2(), Immediate(13, 9))
  rtgtest.JAL(vreg, Immediate(21, 16))
  rtgtest.AUIPC(vreg, Immediate(32, 32))


# MLIR-LABEL: rtg.test @test4_integer_to_immediate()
# MLIR-NEXT: [[V0:%.+]] = rtg.fixed_reg
# MLIR-NEXT: [[V1:%.+]] = index.constant 2
# MLIR-NEXT: [[V2:%.+]] = rtg.isa.int_to_immediate [[V1]] : !rtg.isa.immediate<12>
# MLIR-NEXT: rtgtest.rv32i.addi [[V0]], [[V0]], [[V2]]


@test()
def test4_integer_to_immediate():
  rtgtest.ADDI(IntegerRegister.t0(), IntegerRegister.t0(),
               Immediate(12, Integer(2)))


# MLIR-LABEL: rtg.test @test7_bools
# MLIR: index.bool.constant false
# MLIR: index.bool.constant true
# MLIR: index.cmp eq(%a, %b)
# MLIR: index.cmp ne(%a, %b)
# MLIR: index.cmp ult(%a, %b)
# MLIR: index.cmp ugt(%a, %b)
# MLIR: index.cmp ule(%a, %b)
# MLIR: index.cmp uge(%a, %b)


@sequence(Bool.type())
def consumer(b):
  pass


@test(("a", Integer.type()), ("b", Integer.type()))
def test7_bools(a, b):
  consumer(Bool(True))
  consumer(Bool(False))
  consumer(a == b)
  consumer(a != b)
  consumer(a < b)
  consumer(a > b)
  consumer(a <= b)
  consumer(a >= b)


# MLIR-LABEL: rtg.test @test8_random_integer
# MLIR-NEXT: rtg.random_number_in_range [%a, %b)


@sequence(Integer.type())
def int_consumer(b):
  pass


@test(("a", Integer.type()), ("b", Integer.type()))
def test8_random_integer(a, b):
  int_consumer(Integer.random(a, b))
