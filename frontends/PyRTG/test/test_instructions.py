# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s

from pyrtg import test, config, Config, IntegerRegister, IntegerRegisterType, Immediate, ImmediateType, instruction, SideEffect, rtgtest, Set


@config
class InstructionTestConfig(Config):
  pass


def mock_return_val_func(typ):
  if isinstance(typ, IntegerRegisterType):
    return IntegerRegister.virtual()
  return None


@instruction(return_val_func=mock_return_val_func,
             args=[(IntegerRegisterType(), SideEffect.WRITE),
                   (IntegerRegisterType(), SideEffect.READ),
                   (ImmediateType(12), SideEffect.READ)])
def ADD1(rd, rs1, imm):
  rtgtest.ADDI(rd, rs1, imm)


@instruction(return_val_func=mock_return_val_func,
             args=[(IntegerRegisterType(), SideEffect.READ_WRITE),
                   (IntegerRegisterType(), SideEffect.READ)])
def ADD2(rd_rs1, rs2):
  rtgtest.ADDI(rd_rs1, rs2, Immediate(12, 42))


@instruction(return_val_func=mock_return_val_func,
             args=[(IntegerRegisterType(), SideEffect.WRITE),
                   (IntegerRegisterType(), SideEffect.WRITE),
                   (IntegerRegisterType(), SideEffect.READ)])
def ADD3(rd1, rd2, rs):
  rtgtest.ADD(rd1, rd2, rs)


# CHECK-LABEL: rtg.test @test_instruction
@test(InstructionTestConfig)
def test_instruction(config):
  rs1 = IntegerRegister.t0()
  imm = Immediate(12, 42)

  # CHECK: rtgtest.addi %
  rd = ADD1(rs1, imm)
  assert rd is not None

  # CHECK: rtgtest.addi %
  rd = IntegerRegister.virtual()
  result = ADD1(rd, rs1, imm)
  assert result is rd

  # CHECK: rtgtest.add %
  results = ADD3(rs1)
  assert isinstance(results, tuple)
  assert len(results) == 2


# CHECK-LABEL: rtg.test @test_side_effect_methods
@test(InstructionTestConfig)
def test_side_effect_methods(config):
  # Test num_read_effects
  assert ADD1.num_read_effects() == 2
  assert ADD2.num_read_effects() == 2
  assert ADD3.num_read_effects() == 1

  # Test num_write_effects
  assert ADD1.num_write_effects() == 1
  assert ADD2.num_write_effects() == 1
  assert ADD3.num_write_effects() == 2

  # Test get_read_arg_types
  read_types_add1 = ADD1.get_read_arg_types()
  assert len(read_types_add1) == 2
  assert read_types_add1[0] == IntegerRegisterType()
  assert read_types_add1[1] == ImmediateType(12)

  read_types_add2 = ADD2.get_read_arg_types()
  assert len(read_types_add2) == 2
  assert read_types_add2[0] == IntegerRegisterType()
  assert read_types_add2[1] == IntegerRegisterType()

  # Test get_write_arg_types
  write_types_add2 = ADD2.get_write_arg_types()
  assert len(write_types_add2) == 1
  assert write_types_add2[0] == IntegerRegisterType()

  write_types_add3 = ADD3.get_write_arg_types()
  assert len(write_types_add3) == 2
  assert write_types_add3[0] == IntegerRegisterType()
  assert write_types_add3[1] == IntegerRegisterType()


# CHECK-LABEL: rtg.test @test_instruction_read_write_operand
@test(InstructionTestConfig)
def test_instruction_read_write_operand(config):
  rd_rs1 = IntegerRegister.t0()
  rs2 = IntegerRegister.t1()

  result = ADD2(rd_rs1, rs2)
  assert result is rd_rs1

  try:
    ADD2(rs2)
    assert False, "Expected ValueError"
  except ValueError as e:
    assert str(
        e) == "Expected 2 arguments (excluding WRITE operands), but got 1"


# CHECK-LABEL: rtg.test @test_instruction_as_seq
@test(InstructionTestConfig)
def test_instruction_as_seq(config):
  reg = IntegerRegister.virtual()
  imm = Immediate(12, 42)

  # CHECK: [[IMM:%.+]] = rtg.constant #rtg.isa.immediate<12, 42>
  # CHECK: [[REG:%.+]] = rtg.virtual_reg
  # CHECK: [[SEQ:%.+]] = rtg.get_sequence @ADD1 : !rtg.sequence<!rtgtest.ireg, !rtgtest.ireg, !rtg.isa.immediate<12>>
  # CHECK: [[SET:%.+]] = rtg.set_create [[SEQ]], [[SEQ]] :
  # CHECK: [[SELECTED:%.+]] = rtg.set_select_random [[SET]] :
  # CHECK: [[SUBST:%.+]] = rtg.substitute_sequence [[SELECTED]]([[REG]], [[REG]], [[IMM]]) :
  # CHECK: [[RAND:%.+]] = rtg.randomize_sequence [[SUBST]]
  # CHECK: rtg.embed_sequence [[RAND]]
  instr_set = Set.create(ADD1, ADD1)
  selected_instr = instr_set.get_random()
  selected_instr(reg, reg, imm)


@instruction(return_val_func=mock_return_val_func,
             args=[(IntegerRegisterType(), SideEffect.WRITE),
                   (IntegerRegisterType(), SideEffect.READ)],
             mnemonic="add",
             extension="some_ext")
def add_with_metadata(rd, rs):
  rtgtest.ADDI(rd, rs, Immediate(12, 0))


# CHECK-LABEL: rtg.test @test_instruction_metadata
@test(InstructionTestConfig)
def test_instruction_metadata(config):
  assert add_with_metadata.get_mnemonic() == "add"
  assert add_with_metadata.get_extension() == "some_ext"
  assert ADD1.get_mnemonic() is None
  assert ADD1.get_extension() is None
