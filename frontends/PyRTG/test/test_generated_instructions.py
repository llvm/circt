# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s

from pyrtg import test, config, Config, IntegerRegister, Immediate
from pyrtg.rtgtest_instructions import add, addi, sub, and_, andi, or_, ori, xor, xori, sll, slli


@config
class GeneratedInstructionTestConfig(Config):
  pass


# CHECK-LABEL: rtg.test @test_generated_add_instructions
@test(GeneratedInstructionTestConfig)
def test_generated_add_instructions(config):
  # Test ADD instruction - automatically allocates destination register
  rs1 = IntegerRegister.t0()
  rs2 = IntegerRegister.t1()

  # CHECK: rtgtest.rv32i.add %
  rd = add(rs1, rs2)
  assert rd is not None

  # Test ADD instruction - with explicit destination register
  rd_explicit = IntegerRegister.virtual()
  # CHECK: rtgtest.rv32i.add %
  result = add(rd_explicit, rs1, rs2)
  assert result is None


# CHECK-LABEL: rtg.test @test_generated_addi_instruction
@test(GeneratedInstructionTestConfig)
def test_generated_addi_instruction(config):
  # Test ADDI instruction with immediate
  rs = IntegerRegister.t0()
  imm = Immediate(12, 42)

  # CHECK: rtgtest.rv32i.addi %
  rd = addi(rs, imm)
  assert rd is not None


# CHECK-LABEL: rtg.test @test_generated_sub_instructions
@test(GeneratedInstructionTestConfig)
def test_generated_sub_instructions(config):
  # Test SUB instruction
  rs1 = IntegerRegister.t0()
  rs2 = IntegerRegister.t1()

  # CHECK: rtgtest.rv32i.sub %
  rd = sub(rs1, rs2)
  assert rd is not None


# CHECK-LABEL: rtg.test @test_generated_shift_instructions
@test(GeneratedInstructionTestConfig)
def test_generated_shift_instructions(config):
  # Test SLL (shift left logical) instruction
  rs1 = IntegerRegister.t0()
  rs2 = IntegerRegister.t1()

  # CHECK: rtgtest.rv32i.sll %
  rd = sll(rs1, rs2)
  assert rd is not None

  # Test SLLI (shift left logical immediate) instruction
  imm = Immediate(5, 3)
  # CHECK: rtgtest.rv32i.slli %
  rd_slli = slli(rs1, imm)
  assert rd_slli is not None


# CHECK-LABEL: rtg.test @test_generated_logical_instructions
@test(GeneratedInstructionTestConfig)
def test_generated_logical_instructions(config):
  rs1 = IntegerRegister.t0()
  rs2 = IntegerRegister.t1()
  imm = Immediate(12, 255)

  # CHECK: rtgtest.rv32i.and %
  rd_and = and_(rs1, rs2)
  assert rd_and is not None

  # CHECK: rtgtest.rv32i.andi %
  rd_andi = andi(rs1, imm)
  assert rd_andi is not None

  # CHECK: rtgtest.rv32i.or %
  rd_or = or_(rs1, rs2)
  assert rd_or is not None

  # CHECK: rtgtest.rv32i.ori %
  rd_ori = ori(rs1, imm)
  assert rd_ori is not None

  # CHECK: rtgtest.rv32i.xor %
  rd_xor = xor(rs1, rs2)
  assert rd_xor is not None

  # CHECK: rtgtest.rv32i.xori %
  rd_xori = xori(rs1, imm)
  assert rd_xori is not None


# CHECK-LABEL: rtg.test @test_generated_instruction_chaining
@test(GeneratedInstructionTestConfig)
def test_generated_instruction_chaining(config):
  # Test chaining multiple generated instructions
  rs1 = IntegerRegister.t0()
  rs2 = IntegerRegister.t1()
  imm = Immediate(12, 5)

  # CHECK: rtgtest.rv32i.add %
  temp1 = add(rs1, rs2)

  # CHECK: rtgtest.rv32i.addi %
  temp2 = addi(temp1, imm)

  # CHECK: rtgtest.rv32i.and %
  result = and_(temp2, rs1)

  assert result is not None
