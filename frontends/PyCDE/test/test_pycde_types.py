# RUN: %PYTHON% %s | FileCheck %s

from pycde import dim, Input, Output, generator, System, Module
from pycde.types import (Bit, Bits, List, SInt, StructType, TypeAlias,
                         UnionType, UInt, Window)
from pycde.testing import unittestmodule
from pycde.signals import Struct, UIntSignal

# CHECK: [('foo', Bits<1>), ('bar', Bits<13>)]
st1 = StructType({"foo": Bit, "bar": Bits(13)})
print(st1.fields)
# CHECK: Bits<1>
print(st1.foo)

array1 = dim(UInt(6))
# CHECK: UInt<6>
print(array1)

array2 = Bits(6) * 10 * 12
# CHECK: Bits<6>[10][12]
print(array2)

int_alias = TypeAlias(Bits(8), "myname1")
# CHECK: myname1
print(int_alias)
assert int_alias == TypeAlias(Bits(8), "myname1")

# CHECK: struct { a: Bits<1>, b: SInt<1>}
struct = StructType({"a": Bit, "b": SInt(1)})
print(struct)

dim_alias = dim(1, 8, name="myname5")

# CHECK: List<Bits<5>>
i5list = List(Bits(5))
print(i5list)


class Dummy(Module):
  pass


# CHECK: hw.type_scope @pycde
# CHECK: hw.typedecl @myname1 : i8
# CHECK: hw.typedecl @myname5 : !hw.array<8xi1>
# CHECK-NOT: hw.typedecl @myname1
# CHECK-NOT: hw.typedecl @myname5
m = System(Dummy)
TypeAlias.declare_aliases(m)
TypeAlias.declare_aliases(m)
m.print()

assert Bit == Bits(1)


class ExStruct(Struct):
  a: Bits(4)
  b: UInt(32)

  def get_b_plus1(self) -> UIntSignal:
    return self.b + 1


print(ExStruct)


# CHECK-LABEL:  hw.module @TestStruct(in %inp1 : !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>, out out1 : ui33, out out2 : !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>)
# CHECK-NEXT:     %b = hw.struct_extract %inp1["b"] {sv.namehint = "inp1__b"} : !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>
# CHECK-NEXT:     [[r0:%.+]] = hwarith.constant 1 : ui1
# CHECK-NEXT:     [[r1:%.+]] = hwarith.add %b, [[r0]] : (ui32, ui1) -> ui33
# CHECK-NEXT:     %a = hw.struct_extract %inp1["a"] {sv.namehint = "inp1__a"} : !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>
# CHECK-NEXT:     %b_0 = hw.struct_extract %inp1["b"] {sv.namehint = "inp1__b"} : !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>
# CHECK-NEXT:     [[r2:%.+]] = hwarith.constant 1 : ui1
# CHECK-NEXT:     [[r3:%.+]] = hwarith.add %b_0, [[r2]] : (ui32, ui1) -> ui33
# CHECK-NEXT:     [[r4:%.+]] = hwarith.cast [[r3]] : (ui33) -> ui32
# CHECK-NEXT:     [[r5:%.+]] = hw.struct_create (%a, [[r4]]) : !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>
# CHECK-NEXT:     hw.output [[r1]], [[r5]] : ui33, !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>
@unittestmodule()
class TestStruct(Module):
  inp1 = Input(ExStruct)
  out1 = Output(UInt(33))
  out2 = Output(ExStruct)

  @generator
  def build(self):
    self.out1 = self.inp1.get_b_plus1()
    s = ExStruct(a=self.inp1.a, b=self.inp1.get_b_plus1().as_uint(32))
    assert type(s) is ExStruct._get_value_class()
    self.out2 = s


# CHECK: union { a: Bits<32>, b: Bits<16>}
u = UnionType([("a", Bits(32)), ("b", Bits(16))])
print(u)
# CHECK: [('a', Bits<32>, 0), ('b', Bits<16>, 0)]
print(u.fields)

# CHECK: union { a: Bits<32>, b: Bits<16> offset 32}
u2 = UnionType([("a", Bits(32)), ("b", Bits(16), 32)])
print(u2)
# CHECK: [('a', Bits<32>, 0), ('b', Bits<16>, 32)]
print(u2.fields)


# CHECK-LABEL:  hw.module @TestUnion(in %in1 : !hw.union<a: i32, b: i16>, out outA : i32, out outB : i16, out out1 : !hw.union<a: i32, b: i16>, out out2 : !hw.union<a: i32, b: i16>) attributes {output_file = #hw.output_file<"TestUnion.sv", includeReplicatedOps>} {
# CHECK-NEXT:     [[R0:%.+]] = hw.union_extract %in1["a"] : !hw.union<a: i32, b: i16>
# CHECK-NEXT:     [[R1:%.+]] = hw.union_extract %in1["b"] : !hw.union<a: i32, b: i16>
# CHECK-NEXT:     %c456_i16 = hw.constant 456 : i16
# CHECK-NEXT:     [[R2:%.+]] = hw.union_create "b", %c456_i16 : !hw.union<a: i32, b: i16>
# CHECK-NEXT:     hw.output [[R0]], [[R1]], %in1, [[R2]] : i32, i16, !hw.union<a: i32, b: i16>, !hw.union<a: i32, b: i16>
@unittestmodule()
class TestUnion(Module):
  in1 = Input(u)
  outA = Output(Bits(32))
  outB = Output(Bits(16))
  out1 = Output(u)
  out2 = Output(u)

  @generator
  def build(ports):
    ports.out1 = ports.in1
    ports.outA = ports.in1["a"]
    ports.outB = ports.in1.b
    ports.out2 = u(("b", 456))


# Test Window type with Window.Frame class
pkt_struct = StructType({
    "hdr": Bits(8),
    "payload": Bits(32) * 4,
    "tail": Bits(4)
})

# Test Window.Frame construction and representation
frame1 = Window.Frame("header", ["hdr", ("payload", 4)])
frame2 = Window.Frame("tail", ["tail"])

# CHECK: Frame('header', ['hdr', ('payload', 4)])
print(frame1)
# CHECK: Frame('tail', ['tail'])
print(frame2)

# Test Window creation with Window.Frame objects
window_with_frames = Window("pkt", pkt_struct, [frame1, frame2])
# CHECK: Window<"pkt", struct { hdr: Bits<8>, payload: Bits<32>[4], tail: Bits<4>}, frames=[Frame('header', ['hdr', ('payload', 4)]), Frame('tail', ['tail'])]>
print(window_with_frames)

# Verify window properties
# CHECK: pkt
print(window_with_frames.name)

# Verify frames property returns Window.Frame objects
frames = window_with_frames.frames
assert len(frames) == 2
assert isinstance(frames[0], Window.Frame)
assert isinstance(frames[1], Window.Frame)
# CHECK: Frame('header', ['hdr', ('payload', 4)])
print(frames[0])
# CHECK: Frame('tail', ['tail'])
print(frames[1])


# CHECK-LABEL: hw.module @TestWindowWrap
# CHECK-NEXT:    [[RES:%.+]] = esi.window.wrap %in_union : !esi.window<"pkt", !hw.struct<hdr: i8, payload: !hw.array<4xi32>, tail: i4>, [<"header", [<"hdr">, <"payload", 4>]>, <"tail", [<"tail">]>]
# CHECK-NEXT:    [[R1:%.+]] = esi.window.unwrap %0 : !esi.window<"pkt", !hw.struct<hdr: i8, payload: !hw.array<4xi32>, tail: i4>, [<"header", [<"hdr">, <"payload", 4>]>, <"tail", [<"tail">]>]>
# CHECK-NEXT:    hw.output [[RES]], [[R1]]
@unittestmodule()
class TestWindowWrap(Module):
  in_union = Input(window_with_frames.lowered_type)
  out_window = Output(window_with_frames)
  out_union = Output(window_with_frames.lowered_type)

  @generator
  @staticmethod
  def build(ports):
    window = window_with_frames.wrap(ports.in_union)
    ports.out_window = window
    ports.out_union = window.unwrap()


# Test Window with single unnamed frame
# CHECK: Window<"pkt_single", struct { hdr: Bits<8>, payload: Bits<32>[4], tail: Bits<4>}, frames=[Frame(None, ['hdr', 'tail'])]>
window_single = Window("pkt_single", pkt_struct, [
    Window.Frame(None, ["hdr", "tail"]),
])
print(window_single)

# CHECK: struct { hdr: Bits<8>, tail: Bits<4>}
print(window_single.lowered_type)
