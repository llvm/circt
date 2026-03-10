# RUN: %rtgtool% %s --seed=0 --output-format=asm | FileCheck %s

from pyrtg import test, config, Config, IntegerRegister
from pyrtg.rtgtest_instructions import and_


@config
class GeneratedInstructionTestConfig(Config):
  pass


# CHECK-LABEL: Begin of test 'test_generated_add_instructions
@test(GeneratedInstructionTestConfig)
def test_generated_add_instructions(config):
  rs1 = IntegerRegister.t0()
  rs2 = IntegerRegister.t1()

  # CHECK: and {{.*}}, t0, t1
  rd = and_(rs1, rs2)
  assert isinstance(rd, IntegerRegister)

  rd = IntegerRegister.t2()
  # CHECK: and t2, t0, t1
  result = and_(rd, rs1, rs2)
  assert result is None
