# REQUIRES: esi-cosim, esi-runtime, rtl-sim
# RUN: rm -rf %t
# RUN: mkdir %t && cd %t

# Build the system.
# RUN: %PYTHON% %s %t 2>&1
"""Hardware design for serialization-correctness probes.

This design exercises the on-the-wire layout invariants of the ESI runtime
serializer/deserializer pair against a hardware implementation that is
deliberately picky about byte order, sign extension, struct field order,
sub-byte field packing, and array element order. Each function is small and
self-checking: the host sends a value with distinguishable bytes/fields/bits,
the hardware applies a position-revealing transform, and the host asserts
the exact expected bytes/fields/bits come back. A mismatch in any of those
five invariants between the host serializer and the hardware wire format
yields a wrong (rather than coincidentally correct) answer.
"""

import sys

import pycde.esi as esi
from pycde import AppID, Clock, Module, Reset, System, generator
from esiaccel.bsp import get_bsp
from pycde.constructs import Mux, Wire
from pycde.signals import BitsSignal, Struct
from pycde.types import Bits, Channel, SInt, UInt


class ByteRotate1(Module):
  """Function ``byte_rotate1``: ui64 -> ui64.

  Rotates the input left by one byte position. Sending
  ``0x0102030405060708`` yields ``0x0203040506070801``: the MSB byte wraps
  around to the LSB. Unlike a byte *swap*, a rotate is **not** its own
  inverse, so a host that mis-orders bytes symmetrically on the send and
  receive paths (the classic symmetric-serdes bug that passes loopback)
  produces a wrong-and-distinguishable answer here.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(UInt(64)))
    args = esi.FuncService.get_call_chans(AppID("byte_rotate1"),
                                          arg_type=UInt(64),
                                          result=result_wire)

    ready = Wire(Bits(1))
    arg, valid = args.unwrap(ready)

    # Numerically: result = (arg << 8) | (arg >> 56).
    # In bit-slice terms: the bottom 56 bits of arg become the top 56 bits
    # of the result and the top byte of arg wraps around to the bottom.
    arg_bits = arg.as_bits(64)
    rotated = BitsSignal.concat([arg_bits[0:56], arg_bits[56:64]]).as_uint(64)

    out_chan, out_ready = Channel(UInt(64)).wrap(rotated, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


PatternArray = UInt(8) * 8

# A deliberately asymmetric byte sequence: the wire ordering is unambiguous
# in either direction (no palindrome / no monotone), and every nibble is
# distinct so a single wrong byte is easy to spot in failure messages.
_BYTE_PATTERN = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]


class BytePatternConst(Module):
  """Function ``byte_pattern_const``: ui8 -> array<ui8, 8>.

  Ignores the trigger byte and returns the constant pattern
  ``[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]`` (in PyCDE / MLIR
  index order). The driver consumes the result as **raw wire bytes**
  rather than through the typed deserializer, so any host-side decoding bug
  (including a symmetric one) is taken out of the loop entirely. A test
  failure here points at a mismatch between the runtime's wire convention
  and the spec, not at a SW round-trip.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(PatternArray))
    args = esi.FuncService.get_call_chans(AppID("byte_pattern_const"),
                                          arg_type=UInt(8),
                                          result=result_wire)

    ready = Wire(Bits(1))
    _, valid = args.unwrap(ready)

    pattern = PatternArray([UInt(8)(b) for b in _BYTE_PATTERN])
    out_chan, out_ready = Channel(PatternArray).wrap(pattern, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


class BytePatternEchoEq(Module):
  """Function ``byte_pattern_echo_eq``: array<ui8, 8> -> ui8.

  Compares the incoming 8-byte array against the same constant pattern as
  :class:`BytePatternConst` and returns ``1`` if every element matches,
  ``0`` otherwise. The driver writes **raw wire bytes** straight into the
  arg channel, taking the host-side serializer out of the loop. A test
  failure means the bytes the runtime put on the wire disagree with the
  spec, not that the host SW ``de``serializer also misread them.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(UInt(8)))
    args = esi.FuncService.get_call_chans(AppID("byte_pattern_echo_eq"),
                                          arg_type=PatternArray,
                                          result=result_wire)

    ready = Wire(Bits(1))
    arg, valid = args.unwrap(ready)

    matches = [arg[i] == UInt(8)(b) for i, b in enumerate(_BYTE_PATTERN)]
    all_match = matches[0]
    for m in matches[1:]:
      all_match = all_match & m
    # ``Mux(sel, false_val, true_val)``: when ``sel`` is 1, pick the last arg.
    result = Mux(all_match, UInt(8)(0), UInt(8)(1))

    out_chan, out_ready = Channel(UInt(8)).wrap(result, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


class SignResult(Struct):
  plus_one: SInt(16)
  neg: SInt(16)
  sign_bit: UInt(1)


class SignProbe(Module):
  """Function ``sign_probe``: si16 -> {plus_one, neg, sign_bit}.

  Returns ``arg+1``, ``-arg`` and ``arg<0`` in a single struct. Exercises
  two's-complement addition, negation and MSB extraction at a sub-32-bit
  width. A host that confuses signed/unsigned encoding or sign-extends
  incorrectly will see a wrong ``plus_one`` near the boundaries
  (``INT16_MIN``/``INT16_MAX``).
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(SignResult))
    args = esi.FuncService.get_call_chans(AppID("sign_probe"),
                                          arg_type=SInt(16),
                                          result=result_wire)

    ready = Wire(Bits(1))
    arg, valid = args.unwrap(ready)

    plus_one = (arg + SInt(16)(1)).as_sint(16)
    neg = (-arg).as_sint(16)
    sign_bit = arg.as_bits(16)[15].as_uint(1)
    result = SignResult(plus_one=plus_one, neg=neg, sign_bit=sign_bit)
    out_chan, out_ready = Channel(SignResult).wrap(result, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


class PackStruct(Struct):
  a: UInt(8)
  b: UInt(16)
  c: UInt(8)
  d: UInt(32)


class PackProbe(Module):
  """Function ``pack_probe``: PackStruct -> PackStruct.

  XORs each field with a unique sentinel of the same width. Sentinels
  (``0xA0``, ``0xB000``, ``0xC0``, ``0xD0000000``) make every byte of the
  reply unique and per-field-distinguishable, so a host that swaps fields
  during packing produces a visibly wrong reply rather than a same-bit-width
  false positive. Catches struct-field order and inter-field padding bugs;
  CIRCT structs are LSB-first on the wire.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(PackStruct))
    args = esi.FuncService.get_call_chans(AppID("pack_probe"),
                                          arg_type=PackStruct,
                                          result=result_wire)

    ready = Wire(Bits(1))
    arg, valid = args.unwrap(ready)

    a = (arg["a"].as_bits(8) ^ Bits(8)(0xA0)).as_uint(8)
    b = (arg["b"].as_bits(16) ^ Bits(16)(0xB000)).as_uint(16)
    c = (arg["c"].as_bits(8) ^ Bits(8)(0xC0)).as_uint(8)
    d = (arg["d"].as_bits(32) ^ Bits(32)(0xD0000000)).as_uint(32)
    result = PackStruct(a=a, b=b, c=c, d=d)
    out_chan, out_ready = Channel(PackStruct).wrap(result, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


class BitPackArg(Struct):
  x: UInt(3)
  y: UInt(5)
  z: UInt(4)
  w: UInt(4)


class BitPackResult(Struct):
  # Same widths as BitPackArg, but rotated: the original ``x`` lands in the
  # ``w_field`` slot, etc. Distinct field names so the rotation cannot be
  # confused for an identity transform by accident.
  w_field: UInt(3)
  z_field: UInt(5)
  y_field: UInt(4)
  x_field: UInt(4)


class BitPackProbe(Module):
  """Function ``bit_pack_probe``: BitPackArg -> BitPackResult.

  Returns ``{w, z, y, x}`` packed back into a 16-bit-wide struct that uses
  the same widths as the argument but in reverse field order. Each field has
  a unique width, and the chosen sentinel values used by the test driver
  (``x=0b001``, ``y=0b10001``, ``z=0b1001``, ``w=0b1100``) are unique within
  their own width, so a misaligned shift or wrong field offset produces a
  wrong number rather than a coincidentally-matching one.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(BitPackResult))
    args = esi.FuncService.get_call_chans(AppID("bit_pack_probe"),
                                          arg_type=BitPackArg,
                                          result=result_wire)

    ready = Wire(Bits(1))
    arg, valid = args.unwrap(ready)

    # Move each input field into the output slot whose width matches.
    result = BitPackResult(
        w_field=arg["x"],  # 3 bits
        z_field=arg["y"],  # 5 bits
        y_field=arg["z"],  # 4 bits
        x_field=arg["w"],  # 4 bits
    )
    out_chan, out_ready = Channel(BitPackResult).wrap(result, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


ArrayProbeArg = UInt(8) * 4
ArrayProbeResult = UInt(8) * 4


class ArrayProbe(Module):
  """Function ``array_probe``: array<ui8, 4> -> array<ui8, 4>.

  Returns ``[arg[0]+10, arg[1]+20, arg[2]+30, arg[3]+40]``. CIRCT arrays
  serialize element-reversed on the wire while lists do not, so this
  exercises a serializer path distinct from the list-based tests. A host
  that forgets to reverse on (de)serialize will see, e.g., ``arg=[1,2,3,4]``
  echoed back as ``[41, 32, 23, 14]`` instead of ``[11, 22, 33, 44]``.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(ArrayProbeResult))
    args = esi.FuncService.get_call_chans(AppID("array_probe"),
                                          arg_type=ArrayProbeArg,
                                          result=result_wire)

    ready = Wire(Bits(1))
    arg, valid = args.unwrap(ready)

    # Per-index sentinels (10, 20, 30, 40) make the element order observable.
    # Build the output in increasing-index order; the C++ host driver
    # asserts ``out[i] == arg[i] + 10*(i+1)``.
    plus = [(arg[i] + UInt(8)(10 * (i + 1))).as_uint(8) for i in range(4)]
    out_array = ArrayProbeResult(plus)
    out_chan, out_ready = Channel(ArrayProbeResult).wrap(out_array, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


class Top(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    ByteRotate1(clk=ports.clk,
                rst=ports.rst,
                appid=AppID("byte_rotate1_inst"))
    BytePatternConst(clk=ports.clk,
                     rst=ports.rst,
                     appid=AppID("byte_pattern_const_inst"))
    BytePatternEchoEq(clk=ports.clk,
                      rst=ports.rst,
                      appid=AppID("byte_pattern_echo_eq_inst"))
    SignProbe(clk=ports.clk, rst=ports.rst, appid=AppID("sign_probe_inst"))
    PackProbe(clk=ports.clk, rst=ports.rst, appid=AppID("pack_probe_inst"))
    BitPackProbe(clk=ports.clk,
                 rst=ports.rst,
                 appid=AppID("bit_pack_probe_inst"))
    ArrayProbe(clk=ports.clk, rst=ports.rst, appid=AppID("array_probe_inst"))


if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2] if len(sys.argv) > 2 else None)
  s = System(bsp(Top), name="SerializationProbes", output_directory=sys.argv[1])
  s.compile()
  s.package()
