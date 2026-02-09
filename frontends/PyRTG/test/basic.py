# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s --check-prefix=MLIR
# RUN: %rtgtool% %s --seed=0 --output-format=elaborated | FileCheck %s --check-prefix=ELABORATED
# RUN: %rtgtool% %s --seed=0 -o %t --output-format=asm && FileCheck %s --input-file=%t --check-prefix=ASM

from pyrtg import test, sequence, config, Config, Param, PythonParam, rtg, Label, LabelType, Set, SetType, Integer, IntegerType, Bag, rtgtest, Immediate, ImmediateType, IntegerRegister, Array, ArrayType, Bool, BoolType, Tuple, TupleType, MemoryBlock, Memory, String, report_failure

# MLIR-LABEL: rtg.target @Singleton : !rtg.dict<>
# MLIR-NEXT: }


@config
class Singleton(Config):
  pass


# MLIR-LABEL: rtg.target @Tgt0 : !rtg.dict<entry0: !rtg.set<index>>
# MLIR-NEXT: [[C0:%.+]] = index.constant 0
# MLIR-NEXT: [[C1:%.+]] = index.constant 1
# MLIR-NEXT: [[SET:%.+]] = rtg.set_create [[C0:%.+]], [[C1:%.+]] : index
# MLIR-NEXT: rtg.yield [[SET]] : !rtg.set<index>
# MLIR-NEXT: }


@config
class Tgt0(Config):

  entry0 = Param(loader=lambda: Set.create(Integer(0), Integer(1)))


# MLIR-LABEL: rtg.target @Tgt1 : !rtg.dict<entry0: index, entry1: !rtg.isa.label>
# MLIR-DAG: [[C0:%.+]] = index.constant 0
# MLIR-DAG: [[LBL:%.+]] = rtg.constant #rtg.isa.label<"l0">
# MLIR: rtg.yield [[C0]], [[LBL]] : index, !rtg.isa.label


@config
class Tgt1(Config):

  entry0 = Param(loader=lambda: Integer(0))
  entry1 = Param(loader=lambda: Label.declare("l0"))


# MLIR-LABEL: rtg.target @Tgt2
# MLIR-NEXT: [[V0:%.+]] = rtg.isa.memory_block_declare [0x0 - 0x3f] : !rtg.isa.memory_block<32>
# MLIR-NEXT: rtg.yield [[V0]] : !rtg.isa.memory_block<32>


@config
class Tgt2(Config):

  mem_blk = Param(loader=lambda: MemoryBlock.declare(
      base_address=0, end_address=63, address_width=32))


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
# MLIR-NEXT: [[ARR5:%.+]] = rtg.array_create [[IDX2]], [[IDX1]] : index
# MLIR-NEXT: [[ARR6:%.+]] = rtg.array_inject [[ARR5]][[[IDX1]]], [[IDX3]] : !rtg.array<index>
# MLIR-NEXT: [[SIZE:%.+]] = rtg.array_size [[ARR6]] : !rtg.array<index>
# MLIR-NEXT: rtg.yield [[RES0]], [[ARR1]], [[ARR4]], [[SIZE]] : index, !rtg.array<!rtg.array<index>>, !rtg.array<!rtg.array<index>>, index


@config
class Tgt4(Config):

  arr0 = Param(loader=lambda: Array.create([Integer(
      1), Integer(2), Integer(3)], IntegerType())[2])
  arr1 = Param(loader=lambda: Array.create([], ArrayType(IntegerType())))
  arr2 = Param(loader=lambda: Array.create([
      Array.create([Integer(y * 10 + x) for x in range(3)], IntegerType()) for y
      in range(2)
  ], ArrayType(IntegerType())))
  arr3 = Param(loader=lambda: Array.create([Integer(2), Integer(
      1)], IntegerType()).set(1, Integer(3)).size())


@sequence([SetType(TupleType([IntegerType(), BoolType()]))])
def seq2(set):
  pass


# MLIR-LABEL: rtg.test @test0
# MLIR-NEXT: }

# ELABORATED-LABEL: rtg.test @test0
# ELABORATED-NEXT: }

# ASM-LABEL: Begin of test 'test0
# ASM: End of test 'test0


@test(Singleton)
def test0(config):
  pass


# MLIR-LABEL: rtg.test @test1_args
# MLIR-SAME: (entry0 = [[SET:%.+]]: !rtg.set<index>)
# MLIR-NEXT: [[STR0:%.+]] = rtg.constant "L_" : !rtg.string
# MLIR-NEXT: [[RAND:%.+]] = rtg.set_select_random [[SET]] : !rtg.set<index>
# MLIR-NEXT: [[INT_AS_STR:%.+]] = rtg.int_format [[RAND]]
# MLIR-NEXT: [[STR:%.+]] = rtg.string_concat [[STR0]], [[INT_AS_STR]]
# MLIR-NEXT: [[LBL:%.+]] = rtg.string_to_label [[STR]]
# MLIR-NEXT: rtg.label local [[LBL]]
# MLIR-NEXT: }

# ELABORATED-LABEL: rtg.test @test1_args_Tgt0
# CHECK: [[LBL:%.+]] = rtg.constant #rtg.isa.label<"L_0">
# CHECK-NEXT: rtg.label local [[LBL]]
# CHECK-NEXT: }

# ASM-LABEL: Begin of test 'test1_args
# ASM-NEXT: L_0:
# ASM: End of test 'test1_args


@test(Tgt0)
def test1_args(config):
  i = config.entry0.get_random()
  Label.declare(String("L_") + i.to_string()).place()


# MLIR-LABEL: rtg.test @test2_labels
# MLIR-DAG: index.constant 2
# MLIR-DAG: index.constant 1
# MLIR-DAG: [[L0:%.+]] = rtg.constant #rtg.isa.label<"l0">
# MLIR-DAG: [[STR:%.+]] = rtg.constant "l1" : !rtg.string
# MLIR-DAG: [[LBL5:%.+]] = rtg.constant #rtg.isa.label<"L_5">
# MLIR-DAG: [[LBL3:%.+]] = rtg.constant #rtg.isa.label<"L_3">
# MLIR-NEXT: [[L1:%.+]] = rtg.label_unique_decl [[STR]]
# MLIR-NEXT: [[L2:%.+]] = rtg.label_unique_decl [[STR]]
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

# MLIR-NEXT: rtg.label local [[LBL5]]
# MLIR-NEXT: rtg.label local [[LBL3]]

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
# ELABORATED-DAG: [[L0:%.+]] = rtg.constant #rtg.isa.label<"l0">
# ELABORATED-DAG: [[L1:%.+]] = rtg.constant #rtg.isa.label<"l1_0">
# ELABORATED-DAG: [[L2:%.+]] = rtg.constant #rtg.isa.label<"l1_1">
# ELABORATED-DAG: [[LBL5:%.+]] = rtg.constant #rtg.isa.label<"L_5">
# ELABORATED-DAG: [[LBL3:%.+]] = rtg.constant #rtg.isa.label<"L_3">
# ELABORATED-DAG: [[L5:%.+]] = rtg.constant #rtg.isa.label<"s1">

# ELABORATED-NEXT: rtg.label global [[L0]]
# ELABORATED-NEXT: rtg.label external [[L1]]
# ELABORATED-NEXT: rtg.label local [[L2]]

# ELABORATED-NEXT: rtg.label local [[L0]]
# ELABORATED-NEXT: rtg.label local [[L2]]

# ELABORATED-NEXT: rtg.label local [[LBL5]]
# ELABORATED-NEXT: rtg.label local [[LBL3]]

# ELABORATED-NEXT: rtg.label local
# ELABORATED-NEXT: rtg.label local

# ELABORATED-NEXT: rtg.label local [[L1]]
# ELABORATED-NEXT: rtg.label local [[L1]]
# ELABORATED-NEXT: rtg.label local [[L0]]

# ELABORATED-NEXT: rtg.label local [[L5]]

# ELABORATED-NEXT: }

# ASM-LABEL: Begin of test 'test2_labels
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

# ASM-NEXT: l1_0:
# ASM-NEXT: l1_0:
# ASM-NEXT: l0:

# ASM-NEXT: s1:

# ASM: End of test 'test2_labels


@test(Singleton)
def test2_labels(config):
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
  l3 = Label.declare(String("L_") + add.to_string())
  l3.place()
  l4 = Label.declare("L_3")
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
# MLIR-NEXT: [[V0:%.+]] = rtg.constant #rtg.isa.immediate<32, 286331153> : !rtg.isa.immediate<32>
# MLIR-NEXT: [[V1:%.+]] = rtg.constant #rtg.isa.immediate<32, -2147483648> : !rtg.isa.immediate<32>
# MLIR-NEXT: [[V2:%.+]] = rtg.constant #rtg.isa.immediate<32, 2147483647> : !rtg.isa.immediate<32>
# MLIR-NEXT: %idx2097151 = index.constant 2097151
# MLIR-NEXT: %idx0 = index.constant 0
# MLIR-NEXT: [[IMM32:%.+]] = rtg.constant #rtg.isa.immediate<32, 32>
# MLIR-NEXT: [[IMM21:%.+]] = rtg.constant #rtg.isa.immediate<21, 16>
# MLIR-NEXT: [[IMM13:%.+]] = rtg.constant #rtg.isa.immediate<13, 9>
# MLIR-NEXT: [[T2:%.+]] = rtg.constant #rtgtest.t2 : !rtgtest.ireg
# MLIR-NEXT: [[IMM5:%.+]] = rtg.constant #rtg.isa.immediate<5, 4>
# MLIR-NEXT: [[T1:%.+]] = rtg.constant #rtgtest.t1 : !rtgtest.ireg
# MLIR-NEXT: [[IMM12:%.+]] = rtg.constant #rtg.isa.immediate<12, 8>
# MLIR-NEXT: [[T0:%.+]] = rtg.constant #rtgtest.t0 : !rtgtest.ireg
# MLIR-NEXT: [[VREG:%.+]] = rtg.virtual_reg [#rtgtest.t0 : !rtgtest.ireg, #rtgtest.t1 : !rtgtest.ireg, #rtgtest.t2 : !rtgtest.ireg, #rtgtest.t3 : !rtgtest.ireg, #rtgtest.t4 : !rtgtest.ireg, #rtgtest.t5 : !rtgtest.ireg, #rtgtest.t6 : !rtgtest.ireg, #rtgtest.a7 : !rtgtest.ireg, #rtgtest.a6 : !rtgtest.ireg, #rtgtest.a5 : !rtgtest.ireg, #rtgtest.a4 : !rtgtest.ireg, #rtgtest.a3 : !rtgtest.ireg, #rtgtest.a2 : !rtgtest.ireg, #rtgtest.a1 : !rtgtest.ireg, #rtgtest.a0 : !rtgtest.ireg, #rtgtest.s1 : !rtgtest.ireg, #rtgtest.s2 : !rtgtest.ireg, #rtgtest.s3 : !rtgtest.ireg, #rtgtest.s4 : !rtgtest.ireg, #rtgtest.s5 : !rtgtest.ireg, #rtgtest.s6 : !rtgtest.ireg, #rtgtest.s7 : !rtgtest.ireg, #rtgtest.s8 : !rtgtest.ireg, #rtgtest.s9 : !rtgtest.ireg, #rtgtest.s10 : !rtgtest.ireg, #rtgtest.s11 : !rtgtest.ireg, #rtgtest.s0 : !rtgtest.ireg, #rtgtest.ra : !rtgtest.ireg, #rtgtest.sp : !rtgtest.ireg]
# MLIR-NEXT: rtgtest.rv32i.addi [[VREG]], [[T0]], [[IMM12]]
# MLIR-NEXT: rtgtest.rv32i.slli [[VREG]], [[T1]], [[IMM5]]
# MLIR-NEXT: rtgtest.rv32i.beq [[VREG]], [[T2]], [[IMM13]] : !rtg.isa.immediate<13>
# MLIR-NEXT: rtgtest.rv32i.jal [[VREG]], [[IMM21]] : !rtg.isa.immediate<21>
# MLIR-NEXT: rtgtest.rv32i.auipc [[VREG]], [[IMM32]] : !rtg.isa.immediate<32>
# MLIR-NEXT: [[RND:%.+]] = rtg.random_number_in_range [%idx0, %idx2097151]
# MLIR-NEXT: [[RND_IMM:%.+]] = rtg.isa.int_to_immediate [[RND]]
# MLIR-NEXT: rtgtest.rv32i.jal [[VREG]], [[RND_IMM]] : !rtg.isa.immediate<21>
# MLIR-NEXT: rtgtest.rv32i.auipc [[VREG]], [[V0]] : !rtg.isa.immediate<32>
# MLIR-NEXT: rtgtest.rv32i.auipc [[VREG]], [[V2]] : !rtg.isa.immediate<32>
# MLIR-NEXT: rtgtest.rv32i.auipc [[VREG]], [[V1]] : !rtg.isa.immediate<32>
# MLIR-NEXT: }


@test(Singleton)
def test3_registers_and_immediates(config):
  vreg = IntegerRegister.virtual()
  imm12 = Immediate(12, 8)
  rtgtest.ADDI(vreg, IntegerRegister.t0(), imm12)
  rtgtest.SLLI(vreg, IntegerRegister.t1(), Immediate(5, 4))
  rtgtest.BEQ(vreg, IntegerRegister.t2(), Immediate(13, 9))
  rtgtest.JAL(vreg, Immediate(21, 16))
  rtgtest.AUIPC(vreg, Immediate(32, 32))
  rtgtest.JAL(vreg, Immediate.random(21))
  imm4 = Immediate(4, 1)
  rtgtest.AUIPC(vreg, imm4.replicate(8))
  rtgtest.AUIPC(vreg, Immediate.smax(32))
  rtgtest.AUIPC(vreg, Immediate.smin(32))


# MLIR-LABEL: rtg.test @test4_integer_to_immediate()
# MLIR-NEXT: [[V0:%.+]] = rtg.constant
# MLIR-NEXT: [[V1:%.+]] = index.constant 2
# MLIR-NEXT: [[V2:%.+]] = rtg.isa.int_to_immediate [[V1]] : !rtg.isa.immediate<12>
# MLIR-NEXT: rtgtest.rv32i.addi [[V0]], [[V0]], [[V2]]


@test(Singleton)
def test4_integer_to_immediate(config):
  rtgtest.ADDI(IntegerRegister.t0(), IntegerRegister.t0(),
               Immediate(12, Integer(2)))


# MLIR-LABEL: rtg.test @test6_memories
# MLIR-NEXT: [[REG:%.+]] = rtg.constant #rtgtest.t0 : !rtgtest.ireg
# MLIR-NEXT: [[IDX8:%.+]] = index.constant 8
# MLIR-NEXT: [[IDX4:%.+]] = index.constant 4
# MLIR-NEXT: [[MEM:%.+]] = rtg.isa.memory_alloc %mem_blk, [[IDX8]], [[IDX4]] : !rtg.isa.memory_block<32>
# MLIR-NEXT: [[SIZE:%.+]] = rtg.isa.memory_size [[MEM]] : !rtg.isa.memory<32>
# MLIR-NEXT: [[IMM:%.+]] = rtg.isa.int_to_immediate [[SIZE]] : !rtg.isa.immediate<32>
# MLIR-NEXT: rtgtest.rv32i.auipc [[REG]], [[IMM]] : !rtg.isa.immediate<32>


@test(Tgt2)
def test6_memories(config):
  mem = Memory.alloc(config.mem_blk, size=8, align=4)
  rtgtest.AUIPC(IntegerRegister.t0(), Immediate(32, mem.size()))


# MLIR-LABEL: rtg.test @test7_bools
# MLIR: index.bool.constant false
# MLIR: index.bool.constant true
# MLIR: index.cmp eq(%a, %b)
# MLIR: index.cmp ne(%a, %b)
# MLIR: index.cmp ult(%a, %b)
# MLIR: index.cmp ugt(%a, %b)
# MLIR: index.cmp ule(%a, %b)
# MLIR: index.cmp uge(%a, %b)


@config
class TwoIntegers(Config):
  a = Param(loader=lambda: Integer(0))
  b = Param(loader=lambda: Integer(1))


@sequence([BoolType()])
def consumer(b):
  pass


@test(TwoIntegers)
def test7_bools(config):
  consumer(Bool(True))
  consumer(Bool(False))
  consumer(config.a == config.b)
  consumer(config.a != config.b)
  consumer(config.a < config.b)
  consumer(config.a > config.b)
  consumer(config.a <= config.b)
  consumer(config.a >= config.b)


# MLIR-LABEL: rtg.test @test8_random_integer
# MLIR-NEXT: rtg.random_number_in_range [%a, %b]


@sequence([IntegerType()])
def int_consumer(b):
  pass


@test(TwoIntegers)
def test8_random_integer(config):
  int_consumer(Integer.random(config.a, config.b))


# MLIR-LABEL: rtg.test @test90_tuples
# MLIR-NEXT: [[V0:%.+]] = rtg.tuple_create %a, %b : index, i1
# MLIR-NEXT: rtg.tuple_extract [[V0]] at 1 : !rtg.tuple<index, i1>


@config
class Test90Config(Config):

  a = Param(loader=lambda: Integer(0))
  b = Param(loader=lambda: Bool(True))
  tup = Param(loader=lambda: Tuple.create(Integer(1), Bool(False)))


@test(Test90Config)
def test90_tuples(config):
  tup = Tuple.create(config.a, config.b)
  consumer(tup[1])


# MLIR-LABEL: rtg.test @test91_sets
# MLIR-NEXT: rtg.set_cartesian_product %a, %b : !rtg.set<index>, !rtg.set<i1>
# MLIR: rtg.bag_convert_to_set %c : !rtg.bag<index>
# MLIR: rtg.set_convert_to_bag %a : !rtg.set<index>


@config
class Test91Config(Config):

  a = Param(loader=lambda: Set.create(Integer(0)))
  b = Param(loader=lambda: Set.create(Bool(True)))
  c = Param(loader=lambda: Bag.create((0, Integer(0))))


@test(Test91Config)
def test91_sets(config):
  seq2(Set.cartesian_product(config.a, config.b))
  int_consumer(config.c.to_set().get_random())
  int_consumer(config.a.to_bag().get_random())


# MLIR-LABEL: rtg.target @PythonParams : !rtg.dict<xlen_64: !rtg.tuple>
# MLIR-NEXT: [[TUP:%.+]] = rtg.tuple_create
# MLIR-NEXT: rtg.yield [[TUP]] : !rtg.tuple


@config
class PythonParams(Config):
  xlen = PythonParam(64)


# MLIR-LABEL: rtg.test @test92_python_params
# MLIR-NEXT: [[LBL:%.+]] = rtg.constant #rtg.isa.label<"python_64">
# MLIR-NEXT: rtg.label local [[LBL]]


@test(PythonParams)
def test92_python_params(config):
  Label.declare("python_" + str(config.xlen)).place()


# MLIR-LABEL: rtg.test @test93_immediate_ops
# MLIR-NEXT: [[IMM:%.+]] = rtg.constant #rtg.isa.immediate<4, 4>
# MLIR: rtg.substitute_sequence {{%.+}}([[IMM]]) : !rtg.sequence<!rtg.isa.immediate<4>>


@sequence([ImmediateType(4)])
def immediate_consumer(imm: Immediate):
  pass


@test(Singleton)
def test93_immediate_ops(config):
  imm1 = Immediate(12, 8)
  imm2 = Immediate(8, 4)
  concat = Immediate.concat(imm1, imm2, imm2)
  slice = concat[8:12]
  immediate_consumer(slice)


# MLIR-LABEL: rtg.sequence @seq0
# MLIR-SAME: ([[SET:%.+]]: !rtg.set<!rtg.isa.label>)
# MLIR-NEXT: [[LABEL:%.+]] = rtg.set_select_random [[SET]]
# MLIR-NEXT: rtg.label local [[LABEL]]
# MLIR-NEXT: }


@sequence([SetType(LabelType())])
def seq0(set: Set):
  set.get_random().place()


# MLIR-LABEL: rtg.sequence @seq1
# MLIR-NEXT: [[LABEL:%.+]] = rtg.constant #rtg.isa.label<"s1">
# MLIR-NEXT: rtg.label local [[LABEL]]
# MLIR-NEXT: }


@sequence([])
def seq1():
  Label.declare("s1").place()
