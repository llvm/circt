# RUN: %rtgtool% %s --seed=0 --output-format=asm | FileCheck %s

from pyrtg import test, MemoryBlock, Memory, target, entry, rtgtest, IntegerRegister


@target
class Target:

  @entry
  def mem_blk():
    return MemoryBlock.declare(0, 31, 32)


# CHECK-LABEL: Begin of test0
# CHECK-EMPTY:
# CHECK-NEXT: la t0, 0
# CHECK-NEXT: la t1, 8
# CHECK-EMPTY:
# CHECK-NEXT: End of test0


@test(("mem_blk", MemoryBlock.type(32)))
def test0(mem_blk):
  mem0 = Memory.alloc(mem_blk, 8, 4)
  mem1 = Memory.alloc(mem_blk, 8, 4)

  rtgtest.LA(IntegerRegister.t0(), mem0)
  rtgtest.LA(IntegerRegister.t1(), mem1)
