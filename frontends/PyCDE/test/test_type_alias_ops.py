# RUN: %PYTHON% %s | FileCheck %s

import operator

from pycde import Clock, Input, Module, generator
from pycde.constructs import If, Mux
from pycde.dialects import comb, hwarith
from pycde.signals import And, BitsSignal, Or
from pycde.testing import unittestmodule
from pycde.types import Array, Bit, Bits, SInt, StructType, TypeAlias, UInt

Word = TypeAlias(Bits(8), "word")
NestedWord = TypeAlias(Word, "nested_word")
Flag = TypeAlias(Bit, "flag")
Index = TypeAlias(Bits(2), "index")
BitIndex = TypeAlias(Bits(3), "bit_index")
Unsigned = TypeAlias(UInt(8), "unsigned")
Signed = TypeAlias(SInt(8), "signed")
NestedSigned = TypeAlias(Signed, "nested_signed")
Record = TypeAlias(StructType({"data": Word}), "record")


# CHECK-LABEL: hw.module @AliasedBits
# CHECK: hw.bitcast %lhs {{.*}} -> i8
# CHECK: hw.bitcast %rhs {{.*}} -> i8
# CHECK: comb.and bin {{.*}} : i8
# CHECK: comb.icmp bin eq {{.*}} : i8
# CHECK: comb.extract {{.*}} : (i8) -> i4
# CHECK: comb.concat {{.*}} : i8, i8
# CHECK: comb.shru bin {{.*}} : i8
# CHECK: comb.truth_table
@unittestmodule(run_passes=True)
class AliasedBits(Module):
  lhs = Input(Word)
  rhs = Input(NestedWord)
  plain = Input(Bits(8))
  index = Input(BitIndex)
  flag = Input(Flag)

  @generator
  def build(ports):
    lhs, rhs, plain = ports.lhs, ports.rhs, ports.plain
    for a, b in ((lhs, rhs), (lhs, plain), (plain, rhs)):
      for op in (operator.and_, operator.or_, operator.xor):
        assert op(a, b).type == Bits(8)
      assert (~a).type == Bits(8)
      for op in (operator.eq, operator.ne):
        assert op(a, b).type == Bit

    assert (lhs & rhs).name == "lhs_and_rhs"
    assert lhs[:4].type == Bits(4)
    assert lhs[0].type == Bit
    assert lhs[:0].type == Bits(0)
    assert BitsSignal.concat([lhs, rhs]).type == Bits(16)
    assert lhs[ports.index].type == Bit
    assert lhs.slice(ports.index, 3).type == Bits(3)
    assert lhs.pad_or_truncate(4).type == Bits(4)
    assert lhs.pad_or_truncate(12).type == Bits(12)
    assert lhs.and_reduce().type == Bit
    assert lhs.or_reduce().type == Bit
    assert And(lhs, rhs, plain).type == Bits(8)
    assert Or(lhs, rhs, plain).type == Bits(8)

    assert plain.as_bits() is plain
    assert lhs.as_bits().type == Bits(8)
    assert rhs.as_bits(8).type == Bits(8)
    assert lhs.as_uint().type == UInt(8)
    assert lhs.as_sint().type == SInt(8)
    assert rhs.as_uint(4).type == UInt(4)
    assert rhs.as_sint(12).type == SInt(12)

    # The dialect builders also accept raw values and keyword operands.
    assert comb.AndOp(lhs.value, rhs, plain.value).type == Bits(8)
    assert comb.EqOp(lhs=lhs.value, rhs=rhs).type == Bit
    for op in (comb.AddOp, comb.MulOp, comb.SubOp, comb.DivSOp, comb.DivUOp,
               comb.ModSOp, comb.ModUOp, comb.ShlOp, comb.ShrSOp, comb.ShrUOp):
      assert op(lhs.value, rhs.value).type == Bits(8)
    assert comb.ShrUOp(lhs=lhs, rhs=rhs.value).type == Bits(8)
    assert comb.ParityOp(lhs).type == Bit
    assert comb.ReplicateOp(Bits(16), rhs.value).type == Bits(16)
    assert comb.ReverseOp(lhs).type == Bits(8)
    assert comb.TruthTableOp([ports.flag, ports.flag.value],
                             [False, True, True, False]).type == Bit


# CHECK-LABEL: hw.module @AliasedIntegers
# CHECK: hw.bitcast %signed {{.*}} -> si8
# CHECK: hw.bitcast %unsigned {{.*}} -> ui8
# CHECK: hwarith.add {{.*}} : (si8, ui8) -> si10
# CHECK: hwarith.icmp eq {{.*}} : si8, ui8
# CHECK: hwarith.cast {{.*}} : (si8) -> i12
@unittestmodule(run_passes=True)
class AliasedIntegers(Module):
  signed = Input(NestedSigned)
  unsigned = Input(Unsigned)

  @generator
  def build(ports):
    signed, unsigned = ports.signed, ports.unsigned
    for a, b in ((signed, unsigned), (signed, UInt(8)(3)), (SInt(8)(-2),
                                                            unsigned)):
      assert (a + b).type == SInt(10)
      assert (a - b).type == SInt(10)
      assert (a * b).type == SInt(16)
      assert (a / b).type == SInt(8)
      for op in (operator.eq, operator.ne, operator.lt, operator.le,
                 operator.gt, operator.ge):
        assert op(a, b).type == Bit

    assert (signed + unsigned).name == "signed_plus_unsigned"
    assert (unsigned + unsigned).type == UInt(9)
    assert (unsigned - unsigned).type == SInt(9)
    assert (unsigned * unsigned).type == UInt(16)
    assert (unsigned / unsigned).type == UInt(8)
    assert (-signed).type == SInt(16)
    assert (signed + 1).type == SInt(9)
    assert (unsigned + 1).type == UInt(9)
    assert (unsigned == 1).type == Bit

    assert signed.as_sint().type == SInt(8)
    assert unsigned.as_uint().type == UInt(8)
    assert signed.as_bits(12).type == Bits(12)
    assert unsigned.as_sint(12).type == SInt(12)
    assert signed.as_uint(4).type == UInt(4)
    assert hwarith.AddOp(lhs=signed.value, rhs=unsigned).type == SInt(10)
    assert hwarith.ICmpOp(0, signed.value, unsigned.value).type == Bit
    assert hwarith.CastOp(unsigned.value, Bits(8)).type == Bits(8)


# CHECK-LABEL: hw.module @AliasedSelectors
# CHECK: hw.bitcast %sel {{.*}} -> i1
# CHECK: comb.mux bin {{.*}} : !hw.typealias<@pycde::@word, i8>
# CHECK: comb.mux bin {{.*}} : !hw.typealias<@pycde::@record,
# CHECK: hw.bitcast %index {{.*}} -> i2
# CHECK: hw.array_get {{.*}} : !hw.array<3xtypealias<@pycde::@word, i8>>, i2
# CHECK: comb.and bin {{.*}} : i1
# CHECK: sv.if
@unittestmodule(run_passes=True)
class AliasedSelectors(Module):
  clk = Clock()
  sel = Input(Flag)
  index = Input(Index)
  word = Input(Word)
  record = Input(Record)
  flags = Input(Array(Flag, 3))

  @generator
  def build(ports):
    word, record = ports.word, ports.record
    assert Mux(ports.sel, word, word).type == Word
    assert If(ports.sel, word, word).type == Word
    assert comb.MuxOp(cond=ports.sel.value,
                      trueValue=word,
                      falseValue=word.value).type == Word
    assert Mux(ports.sel, record, record).type == Record
    assert Mux(ports.index, word, word, word).type == Word
    assert ports.flags[ports.index].type == Flag
    assert ports.flags.slice(ports.index, 1).type == Array(Flag, 1)
    assert ports.flags.and_reduce().type == Bit
    assert ports.flags.or_reduce().type == Bit
    assert (record.data & word).type == Bits(8)
    assert word.reg(ports.clk).type == Word
    assert Word(3).type == Word
    assert Unsigned(3).type == Unsigned
    assert Signed(-2).type == Signed
    ports.sel.when_true(lambda: ~ports.sel, clk=ports.clk)
