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
                   (IntegerRegisterType(), SideEffect.READ)])
def ADD1(rd, rs1):
  rtgtest.TwoRegisterInstrOp(rd, rs1)


@instruction(return_val_func=mock_return_val_func,
             args=[(IntegerRegisterType(), SideEffect.READ_WRITE),
                   (IntegerRegisterType(), SideEffect.READ)])
def ADD2(rd_rs1, rs2):
  rtgtest.TwoRegisterInstrOp(rd_rs1, rs2)


@instruction(return_val_func=mock_return_val_func,
             args=[(IntegerRegisterType(), SideEffect.WRITE),
                   (IntegerRegisterType(), SideEffect.WRITE),
                   (IntegerRegisterType(), SideEffect.READ)])
def ADD3(rd1, rd2, rs):
  rtgtest.ThreeRegisterInstrOp(rd1, rd2, rs)


# CHECK-LABEL: rtg.test @test_instruction
@test(InstructionTestConfig)
def test_instruction(config):
  rs1 = IntegerRegister.t0()

  # CHECK: rtgtest.two_register_instr %
  rd = ADD1(rs1)
  assert rd is not None

  # CHECK: rtgtest.two_register_instr %
  rd = IntegerRegister.virtual()
  result = ADD1(rd, rs1)
  assert result is None

  # CHECK: rtgtest.three_register_instr %
  results = ADD3(rs1)
  assert isinstance(results, list)
  assert len(results) == 2

  assert ADD1.num_read_effects() == 1
  assert ADD2.num_read_effects() == 2
  assert ADD3.num_read_effects() == 1


# CHECK-LABEL: rtg.test @test_instruction_read_write_operand
@test(InstructionTestConfig)
def test_instruction_read_write_operand(config):
  rd_rs1 = IntegerRegister.t0()
  rs2 = IntegerRegister.t1()

  result = ADD2(rd_rs1, rs2)
  assert result is None

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

  # CHECK: [[REG:%.+]] = rtg.virtual_reg
  # CHECK: [[SEQ:%.+]] = rtg.get_sequence @ADD1 : !rtg.sequence<!rtgtest.ireg, !rtgtest.ireg>
  # CHECK: [[SET:%.+]] = rtg.set_create [[SEQ]], [[SEQ]] :
  # CHECK: [[SELECTED:%.+]] = rtg.set_select_random [[SET]] :
  # CHECK: [[SUBST:%.+]] = rtg.substitute_sequence [[SELECTED]]([[REG]], [[REG]]) :
  # CHECK: [[RAND:%.+]] = rtg.randomize_sequence [[SUBST]]
  # CHECK: rtg.embed_sequence [[RAND]]
  instr_set = Set.create(ADD1, ADD1)
  selected_instr = instr_set.get_random()
  selected_instr(reg, reg)
