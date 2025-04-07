# RUN: %rtgtool% %s --seed=0 --output-format=asm | FileCheck %s

from pyrtg import test, MemoryBlock, Memory, config, Config, Param, rtgtest, IntegerRegister


@config
class Target(Config):

  mem_blk = Param(loader=lambda: MemoryBlock.declare(32, 0, 32))


# CHECK-LABEL: Begin of test0
# CHECK-EMPTY:
# CHECK-NEXT: la t0, 0
# CHECK-NEXT: la t1, 8
# CHECK-EMPTY:
# CHECK-NEXT: End of test0


@test(Target)
def test0(config):
  mem0 = Memory.alloc(config.mem_blk, 8, 4)
  mem1 = Memory.alloc(config.mem_blk, 8, 4)

  rtgtest.LA(IntegerRegister.t0(), mem0)
  rtgtest.LA(IntegerRegister.t1(), mem1)
