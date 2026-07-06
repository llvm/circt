"""Hardware design for the codegen + port-kind coverage integration test.

Where ``serialization_probes.py`` exercises wire-format invariants, this design
exercises the *port-kind* surface area of the ESI runtime + facade codegen.
Each probe module here is named for the codegen / runtime path it exercises
so a regression in any single path lights up exactly one driver assertion.
"""

import sys

import pycde.esi as esi
from pycde import AppID, Clock, Module, Reset, System, generator
from esiaccel.bsp import get_bsp
from pycde.common import Constant
from pycde.constructs import ControlReg, Counter, Reg, Wire
from pycde.esi import ListWindowToParallel, ListWindowToSerial
from pycde.signals import Struct
from pycde.types import (Array, Bits, Bundle, BundledChannel, Channel,
                         ChannelDirection, ChannelSignaling, List, SInt,
                         StructType, TypeAlias, UInt, Window)

# Custom service declarations for the custom-`@esi.ServiceDecl` raw-channel
# probe. Provides both Bits(8) and Bits(0) (void) channel widths so the
# zero-bit elaboration path is exercised even though the C++ driver only
# drives the byte path today.
SendI8 = Bundle([BundledChannel("send", ChannelDirection.FROM, Bits(8))])
RecvI8 = Bundle([BundledChannel("recv", ChannelDirection.TO, Bits(8))])
SendI0 = Bundle([BundledChannel("send", ChannelDirection.FROM, Bits(0))])
RecvI0 = Bundle([BundledChannel("recv", ChannelDirection.TO, Bits(0))])


@esi.ServiceDecl
class HostComms:
  Send = SendI8
  Recv = RecvI8


@esi.ServiceDecl
class VoidComms:
  Send = SendI0
  Recv = RecvI0


class TypedFuncMultiArg(Module):
  """Typed function with a multi-field argument struct.

  Exercises ``TypedFunction``'s emplace-style ``call(...)`` overload, which
  forwards its arguments into the generated arg struct's constructor so the
  C++ driver can call ``connected->call(a, b)`` instead of building the
  struct itself. The body computes ``a * b`` and returns it as ``ui32``.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(UInt(32)))

    class Args(Struct):
      a: UInt(32)
      b: UInt(32)

    args = esi.FuncService.get_call_chans(AppID("call"),
                                          arg_type=Args,
                                          result=result_wire)
    ready = Wire(Bits(1))
    arg, valid = args.unwrap(ready)
    product = (arg["a"] * arg["b"]).as_uint(32)
    out_chan, out_ready = Channel(UInt(32)).wrap(product, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


class TypedFuncVoidArg(Module):
  """Typed function with a void argument (typed-result specialization).

  The C++ driver invokes ``connected->call().get()`` and asserts the constant
  token comes back. Hardware sends ``0xCAFEF00D`` on every call so a wrong-
  byte-order bug in the result path fails distinguishably.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(UInt(32)))
    args = esi.FuncService.get_call_chans(AppID("call"),
                                          arg_type=Bits(0),
                                          result=result_wire)
    ready = Wire(Bits(1))
    _, valid = args.unwrap(ready)
    token = UInt(32)(0xCAFEF00D)
    out_chan, out_ready = Channel(UInt(32)).wrap(token, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


class AckArgs(Struct):
  tag: UInt(8)
  seq: UInt(16)


class TypedFuncVoidResult(Module):
  """Typed function with a void result (typed-arg specialization).

  Hardware accepts the request and returns a one-byte zero (the void-result
  wire encoding). No state is observable other than that the call completes;
  the test asserts the future resolves without throwing.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(Bits(0)))
    args = esi.FuncService.get_call_chans(AppID("call"),
                                          arg_type=AckArgs,
                                          result=result_wire)
    ready = Wire(Bits(1))
    _, valid = args.unwrap(ready)
    out_chan, out_ready = Channel(Bits(0)).wrap(Bits(0)(0), valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


class NotifyArgs(Struct):
  tag: UInt(8)
  payload: UInt(32)


class CallServiceCallback(Module):
  """Hardware-initiated call into the host via `CallService`.

  The trigger is an MMIO write at offset ``0x10`` (whose write data forms
  the ``payload``) so the driver can deterministically time when the
  callback fires. The callback returns no payload (``Bits(0)`` is the void
  encoding); the host-side handler increments a counter so the driver can
  assert it actually ran.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    mmio_bundle = esi.MMIO.read_write(appid=AppID("trigger"))
    data_resp_chan = Wire(Channel(Bits(64)))
    cmd_chan = mmio_bundle.unpack(data=data_resp_chan)["cmd"]

    # Snoop the cmd-channel handshake (xact = valid & ready) combinationally
    # without consuming it. The echo response below drives the actual
    # handshake; back-pressure on the CallService never reaches the MMIO bus
    # because the latch overwrites in place.
    xact, cmd = cmd_chan.snoop_xact()

    # Echo write data back as the read response.
    data_resp_chan.assign(cmd_chan.transform(lambda c: Bits(64)(c.data)))

    # trigger_xact: a cmd was accepted this cycle AND its offset is 0x10.
    # Use it both as the clock enable for the payload latch and as the
    # assert for the callback's valid flag.
    is_trigger = (cmd.offset == UInt(32)(0x10))
    trigger_xact = xact & is_trigger

    # A new trigger overwrites any previous-but-unconsumed payload
    # (one-outstanding semantics, same as a Mailbox).
    data_reg = cmd.data.as_uint(32).reg(clk, rst, ce=trigger_xact)
    notify_args = NotifyArgs(tag=UInt(8)(0xA5), payload=data_reg)

    # ControlReg holds the callback's valid bit: assert on trigger_xact,
    # clear when the consumer takes the message. ControlReg gives asserts
    # priority on same-cycle ties, so a new trigger landing exactly on the
    # consumption cycle correctly keeps valid asserted for the next message.
    cb_consumed = Wire(Bits(1))
    cb_valid = ControlReg(clk,
                          rst,
                          asserts=[trigger_xact],
                          resets=[cb_consumed])
    cb_chan, cb_ready = Channel(NotifyArgs).wrap(notify_args, cb_valid)
    cb_consumed.assign(cb_valid & cb_ready)
    esi.CallService.call(AppID("callback"), cb_chan, Bits(0))


class EventStruct(Struct):
  ts: UInt(64)
  val: SInt(32)


class TypedReadChannelStruct(Module):
  """To-host channel of `EventStruct`: TypedReadPort polling of a struct.

  Hardware pushes a small bounded sequence of distinct events on reset
  release so the driver can read N items and check exact values. Each event
  has ``ts = i+1`` and ``val = -(i+1)`` so an off-by-one or sign bug shows
  up immediately.
  """

  clk = Clock()
  rst = Reset()

  num_events = Constant(UInt(8), 4)

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    # One counter that advances exactly when the to-host channel handshakes
    # (valid & ready). The valid signal is ``count < num_events`` so the
    # stream goes silent after num_events have been delivered.
    ready_wire = Wire(Bits(1))
    increment_wire = Wire(Bits(1))
    counter = Counter(8)(clk=clk,
                         rst=rst,
                         clear=Bits(1)(0),
                         increment=increment_wire,
                         instance_name="event_counter")
    valid = counter.out < TypedReadChannelStruct.num_events.value
    increment_wire.assign(valid & ready_wire)

    one_based = (counter.out + UInt(8)(1)).as_uint(8)
    ts = one_based.as_uint(64)
    # val = -ts as si32 -- small enough to fit and reveals sign-extension
    # bugs on the host side.
    one_based_u32 = one_based.as_uint(32)
    neg = (UInt(32)(0) - one_based_u32).as_sint(32)
    event = EventStruct(ts=ts, val=neg)
    out_chan, out_ready = Channel(EventStruct).wrap(event, valid)
    ready_wire.assign(out_ready)
    esi.ChannelService.to_host(AppID("data"), out_chan)


class TypedWriteChannelByte(Module):
  """From-host channel of ``ui8``: TypedWritePort + MMIO accumulator readback.

  Hardware accepts every byte and XORs each one into a running register; the
  latest XOR-accumulator value is exposed via the ``accumulator`` MMIO read
  region so the driver can assert what it sent actually arrived. An always-
  ready receiver is fine here because the test sends a small known sequence.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    chan = esi.ChannelService.from_host(AppID("data"), Bits(8))
    always_ready = Bits(1)(1)
    data, valid = chan.unwrap(always_ready)

    acc = Reg(Bits(8), clk=clk, rst=rst, rst_value=0, ce=valid, name="cmds_acc")
    acc.assign(acc ^ data)

    # Expose the accumulator on MMIO read so the driver can verify what
    # arrived without needing an answer channel.
    mmio_bundle = esi.MMIO.read(appid=AppID("accumulator"))
    resp_chan = Wire(Channel(Bits(64)))
    addr_chan = mmio_bundle.unpack(data=resp_chan)["offset"]
    resp_chan.assign(addr_chan.transform(lambda _: acc.as_bits(64)))


class MmioReadWrite(Module):
  """MMIO read/write region that loops back the most recent write.

  The test writes a value to offset 0x10 and reads it back; whatever was
  last written at any 8-byte-aligned offset is what the read returns. Stores all
  writes for simplicity.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    mmio_bundle = esi.MMIO.read_write(appid=AppID("region"))
    resp_chan = Wire(Channel(Bits(64)))
    cmd_chan = mmio_bundle.unpack(data=resp_chan)["cmd"]

    cmd_ready = Wire(Bits(1))
    cmd, cmd_valid = cmd_chan.unwrap(cmd_ready)

    write_handshake = cmd_valid & cmd.write
    storage = Reg(Bits(64),
                  clk=clk,
                  rst=rst,
                  rst_value=0,
                  ce=write_handshake,
                  name="regs_storage")
    storage.assign(cmd.data)

    # Reads always echo the most-recently-written value.
    response, resp_ready = Channel(Bits(64)).wrap(storage, cmd_valid)
    cmd_ready.assign(resp_ready)
    resp_chan.assign(response)


class TelemetryMetric(Module):
  """Free-running ``ui64`` cycle counter exposed as a telemetry metric.

  Hardware increments the counter every clock; the host reads it twice and
  asserts the second read is strictly greater than the first (cycle counts
  are monotonic between any two host-visible reads).
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    cycle_cnt = Counter(64)(clk=ports.clk,
                            rst=ports.rst,
                            clear=Bits(1)(0),
                            increment=Bits(1)(1),
                            instance_name="cycleCounter")
    esi.Telemetry.report_signal(ports.clk, ports.rst, AppID("cycleCount"),
                                cycle_cnt.out)


class IndexedFuncGroup(Module):
  """Module exposing an indexed array of typed-function ports.

  Instantiates ``num_entries`` ``FuncService`` ports under the same appid name
  ``call`` with indices 0..N-1 -- the codegen groups them into a single
  ``IndexedPorts<TypedFunction<...>>`` member, which the C++ driver iterates
  over (``connected->call[i]``). Each entry returns ``arg + (i+1)``, so the
  driver can verify it talked to the right index by sending the same arg to
  every entry and comparing replies.
  """

  num_entries = Constant(UInt(8), 3)

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    for i in range(IndexedFuncGroup.num_entries.value):
      addend = i + 1
      result_wire = Wire(Channel(UInt(16)))
      args = esi.FuncService.get_call_chans(AppID("call", i),
                                            arg_type=UInt(16),
                                            result=result_wire)
      ready = Wire(Bits(1))
      arg, valid = args.unwrap(ready)
      sum_ = (arg + UInt(16)(addend)).as_uint(16)
      out_chan, out_ready = Channel(UInt(16)).wrap(sum_, valid)
      ready.assign(out_ready)
      result_wire.assign(out_chan)


class CustomServiceDeclChannel(Module):
  """Custom-`@esi.ServiceDecl` raw-channel loopback.

  Connects ``HostComms.Recv`` -> ``HostComms.Send`` for an 8-bit byte stream
  (host writes, HW echoes back) and ``VoidComms.Recv`` -> ``VoidComms.Send``
  for a zero-bit "tick" stream. Both pairs are exposed via custom service
  decls rather than the standard `ChannelService`. ``Top`` instantiates two
  copies under indexed appids so the test also covers same-name multi-instance
  hierarchy resolution.
  """

  clk = Clock()

  @generator
  def construct(ports):
    data_in = HostComms.Recv(AppID("byte_in")).unpack()["recv"]
    HostComms.Send(AppID("byte_out")).unpack(send=data_in)

    void_in = VoidComms.Recv(AppID("void_in")).unpack()["recv"]
    VoidComms.Send(AppID("void_out")).unpack(send=void_in)


class StructArgs(Struct):
  a: UInt(16)
  b: SInt(8)


class StructResult(Struct):
  x: SInt(8)
  y: SInt(8)


class TypedFuncStruct(Module):
  """Typed function: small struct -> small struct.

  Returns ``{x = b+1, y = b}`` so the host can verify the arithmetic and
  the order of struct fields end-to-end.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(StructResult))
    args = esi.FuncService.get_call_chans(AppID("call"),
                                          arg_type=StructArgs,
                                          result=result_wire)
    ready = Wire(Bits(1))
    arg, valid = args.unwrap(ready)
    b = arg["b"]
    plus_one = (b + SInt(8)(1)).as_sint(8)
    result = StructResult(x=plus_one, y=b)
    out_chan, out_ready = Channel(StructResult).wrap(result, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


class OddInner(Struct):
  p: UInt(8)
  q: SInt(8)
  r: UInt(8) * 2


class OddStruct(Struct):
  a: UInt(12)
  b: SInt(7)
  inner: OddInner


class TypedFuncNestedStruct(Module):
  """Typed function: nested odd-bit-width struct round-trip with arithmetic
  on every field. Each field gets a distinct addend so a swap of any two
  fields fails distinguishably."""

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(OddStruct))
    args = esi.FuncService.get_call_chans(AppID("call"),
                                          arg_type=OddStruct,
                                          result=result_wire)
    ready = Wire(Bits(1))
    arg, valid = args.unwrap(ready)
    a = (arg["a"] + UInt(12)(1)).as_uint(12)
    b = (arg["b"] + SInt(7)(-3)).as_sint(7)
    inner = arg["inner"]
    p = (inner["p"] + UInt(8)(5)).as_uint(8)
    q = (inner["q"] + SInt(8)(2)).as_sint(8)
    r0 = (inner["r"][0] + UInt(8)(1)).as_uint(8)
    r1 = (inner["r"][1] + UInt(8)(2)).as_uint(8)
    new_inner = OddInner(p=p, q=q, r=[r0, r1])
    result = OddStruct(a=a, b=b, inner=new_inner)
    out_chan, out_ready = Channel(OddStruct).wrap(result, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


class TypedFuncSubByteSigned(Module):
  """Typed function: ``si4 -> si4`` identity.

  The driver tests positive, negative, and the si4 boundary values to
  exercise sign extension at a sub-byte width through the typed facade.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(SInt(4)))
    args = esi.FuncService.get_call_chans(AppID("call"),
                                          arg_type=SInt(4),
                                          result=result_wire)
    ready = Wire(Bits(1))
    arg, valid = args.unwrap(ready)
    out_chan, out_ready = Channel(SInt(4)).wrap(arg, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


ArrayArg = SInt(8) * 1
ArrayResult = TypeAlias(SInt(8) * 2, "ArrayResult")


class TypedFuncArrayResult(Module):
  """Typed function with an array result.

  Receives a one-element array and returns a two-element array containing
  the input element and ``input + 1``. Exercises the typed facade's
  ``std::array`` path end-to-end.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_wire = Wire(Channel(ArrayResult))
    args = esi.FuncService.get_call_chans(AppID("call"),
                                          arg_type=ArrayArg,
                                          result=result_wire)
    ready = Wire(Bits(1))
    arg, valid = args.unwrap(ready)
    elem = arg[0]
    plus_one = (elem + SInt(8)(1)).as_sint(8)
    # The C++ driver asserts on the wire bytes so the convention is observable.
    result_array = ArrayResult([elem, plus_one])
    out_chan, out_ready = Channel(ArrayResult).wrap(result_array, valid)
    ready.assign(out_ready)
    result_wire.assign(out_chan)


# Typed function over a windowed list payload. Uses the auto serial<->parallel
# converters from `pycde.esi` so the burst-protocol state machine doesn't
# have to be hand-rolled here. Each call doubles every list element.
_TRANSFORM_LIST_BULK_WIDTH = 16
_TRANSFORM_LIST_ITEMS_PER_FRAME = 1


class TransformListItem(Struct):
  v: Bits(32)


_TRANSFORM_LIST_STRUCT = StructType([("data", List(TransformListItem))])
_transform_list_window = Window.serial_of(_TRANSFORM_LIST_STRUCT,
                                          _TRANSFORM_LIST_BULK_WIDTH,
                                          _TRANSFORM_LIST_ITEMS_PER_FRAME)


class TypedFuncWindowedList(Module):
  """Typed function: ``window<list<si32>> -> window<list<si32>>``.

  Doubles each element of the input list and emits the result as another
  serial-burst windowed list. Driving the burst protocol is delegated to
  `ListWindowToParallel` / `ListWindowToSerial` so this module only has to
  describe the per-element transform.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    result_chan = Wire(Channel(_transform_list_window))
    args = esi.FuncService.get_call_chans(AppID("call"),
                                          arg_type=_transform_list_window,
                                          result=result_chan)

    s2p = ListWindowToParallel(_transform_list_window)(clk=ports.clk,
                                                       rst=ports.rst,
                                                       serial_in=args)
    parallel_in = s2p.parallel_out

    par_ready = Wire(Bits(1))
    par_window, par_valid = parallel_in.unwrap(par_ready)
    par_struct = par_window.unwrap()

    in_item = par_struct["data"]
    last_bit = par_struct["last"]
    in_v = in_item["v"].as_uint(32)
    doubled = (in_v + in_v).as_bits(32)
    out_item = TransformListItem(v=doubled)

    parallel_result_window_type = Window.default_of(_TRANSFORM_LIST_STRUCT)
    parallel_result_struct = parallel_result_window_type.lowered_type({
        "data": out_item,
        "last": last_bit,
    })
    parallel_result_window = parallel_result_window_type.wrap(
        parallel_result_struct)
    parallel_result_chan, par_result_ready = Channel(
        parallel_result_window_type).wrap(parallel_result_window, par_valid)
    par_ready.assign(par_result_ready)

    p2s = ListWindowToSerial(parallel_result_window_type,
                             _TRANSFORM_LIST_BULK_WIDTH,
                             _TRANSFORM_LIST_ITEMS_PER_FRAME,
                             meta_fifo_depth=4)(
                                 clk=ports.clk,
                                 rst=ports.rst,
                                 parallel_in=parallel_result_chan)
    result_chan.assign(p2s.serial_out)


# Window types for the channel-of-window probes. The struct carries a static
# header (`tag`) plus a list payload (`items`) so the probes exercise the
# header+list shape that today is only tested via the `FuncService` path.
_WINDOW_PROBE_TAG = 0xCAFE
_WINDOW_PROBE_ITEMS = [10, 20, 30, 40]
_WINDOW_PROBE_BULK_WIDTH = 16
_WINDOW_PROBE_ITEMS_PER_FRAME = 1
_window_probe_struct = StructType([("tag", Bits(16)),
                                   ("items", List(Bits(32)))])
_window_probe_window = Window.serial_of(_window_probe_struct,
                                        _WINDOW_PROBE_BULK_WIDTH,
                                        _WINDOW_PROBE_ITEMS_PER_FRAME)


class CallbackWindowedList(Module):
  """HW-initiated callback whose argument is a windowed list with header.

  Combines the callback pattern (``CallService.call``) with the serial-burst
  windowed list payload. An MMIO write at offset ``0x10`` arms one burst;
  the HW then sends the same ``{tag=0xCAFE, items=[10,20,30,40]}`` pattern
  used by the channel probes into the host callback. The host handler
  verifies the payload and the callback returns void (``Bits(0)``).
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    # MMIO trigger: write to offset 0x10 arms one burst.
    trigger_bundle = esi.MMIO.read_write(appid=AppID("trigger"))
    resp_chan = Wire(Channel(Bits(64)))
    cmd_chan = trigger_bundle.unpack(data=resp_chan)["cmd"]
    cmd_xact, cmd = cmd_chan.snoop_xact()
    resp_chan.assign(cmd_chan.transform(lambda c: Bits(64)(c.data)))
    trigger_xact = cmd_xact & (cmd.offset == UInt(32)(0x10))

    n_items = len(_WINDOW_PROBE_ITEMS)
    burst_end = Wire(Bits(1))

    armed = ControlReg(clk, rst, asserts=[trigger_xact], resets=[burst_end])

    par_ready = Wire(Bits(1))
    handshake = armed & par_ready
    idx_counter = Counter(2)(clk=clk,
                             rst=rst,
                             clear=burst_end,
                             increment=handshake,
                             instance_name="cb_window_idx")
    idx = idx_counter.out
    last_bit = (idx == UInt(2)(n_items - 1))
    burst_end.assign(handshake & last_bit)

    item_bits = Array(
        Bits(32), len(_WINDOW_PROBE_ITEMS))(_WINDOW_PROBE_ITEMS)[idx.as_bits()]

    parallel_window_type = Window.default_of(_window_probe_struct)
    par_struct = parallel_window_type.lowered_type({
        "tag": Bits(16)(_WINDOW_PROBE_TAG),
        "items": item_bits,
        "last": last_bit,
    })
    par_window = parallel_window_type.wrap(par_struct)
    parallel_chan, parallel_ready = Channel(parallel_window_type).wrap(
        par_window, armed)
    par_ready.assign(parallel_ready)

    p2s = ListWindowToSerial(parallel_window_type, _WINDOW_PROBE_BULK_WIDTH,
                             _WINDOW_PROBE_ITEMS_PER_FRAME,
                             4)(clk=clk, rst=rst, parallel_in=parallel_chan)

    esi.CallService.call(AppID("callback"), p2s.serial_out, Bits(0))


class ChannelWindowedListRead(Module):
  """To-host channel of ``window<{tag, list<si32>}>``.

  Exercises the typed read path for windowed-list-with-header on a raw
  channel (no `TypedFunction` orchestrator on top). The driver writes any
  value to offset ``0x10`` of the ``trigger`` MMIO region to arm one burst;
  the HW then emits exactly one burst (``tag = 0xCAFE`` and the four-element
  list ``[10, 20, 30, 40]``) and goes idle. Free-running emission would
  unboundedly fill the host runtime's polling queue, so each burst is gated
  on an explicit trigger.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    # MMIO trigger: any write to offset 0x10 arms one burst. Snoop the
    # cmd-channel xact and echo the write data as the read response so the
    # MMIO bus is never back-pressured.
    trigger_bundle = esi.MMIO.read_write(appid=AppID("trigger"))
    resp_chan = Wire(Channel(Bits(64)))
    cmd_chan = trigger_bundle.unpack(data=resp_chan)["cmd"]
    cmd_xact, cmd = cmd_chan.snoop_xact()
    resp_chan.assign(cmd_chan.transform(lambda c: Bits(64)(c.data)))
    trigger_xact = cmd_xact & (cmd.offset == UInt(32)(0x10))

    n_items = len(_WINDOW_PROBE_ITEMS)
    burst_end = Wire(Bits(1))

    # ``armed`` is high for the duration of one burst: set on a trigger
    # write, cleared on the cycle the burst's last beat handshakes.
    armed = ControlReg(clk, rst, asserts=[trigger_xact], resets=[burst_end])

    # Per-beat item index. Increments on a beat handshake, clears at the end
    # of the burst so the next trigger starts fresh from index 0.
    par_ready = Wire(Bits(1))
    handshake = armed & par_ready
    idx_counter = Counter(2)(clk=clk,
                             rst=rst,
                             clear=burst_end,
                             increment=handshake,
                             instance_name="window_read_idx")
    idx = idx_counter.out
    last_bit = (idx == UInt(2)(n_items - 1))
    burst_end.assign(handshake & last_bit)
    item_value = Array(
        Bits(32), len(_WINDOW_PROBE_ITEMS))(_WINDOW_PROBE_ITEMS)[idx.as_bits()]

    # Build the parallel beat. ``Window.default_of`` lowers each beat as
    # ``{<static fields>, <one item>, last}``.
    parallel_window_type = Window.default_of(_window_probe_struct)
    par_struct = parallel_window_type.lowered_type({
        "tag": Bits(16)(_WINDOW_PROBE_TAG),
        "items": item_value,
        "last": last_bit,
    })
    par_window = parallel_window_type.wrap(par_struct)
    parallel_chan, parallel_ready = Channel(parallel_window_type).wrap(
        par_window, armed)
    par_ready.assign(parallel_ready)

    p2s = ListWindowToSerial(parallel_window_type, _WINDOW_PROBE_BULK_WIDTH,
                             _WINDOW_PROBE_ITEMS_PER_FRAME,
                             4)(clk=clk, rst=rst, parallel_in=parallel_chan)
    esi.ChannelService.to_host(AppID("data"), p2s.serial_out)


# Multi-burst read: the list is longer than the serial encoder's data FIFO, so
# `ListWindowToSerial` emits it as several header/data bursts (via its
# split-on-full path) terminated by a single count==0 footer, rather than one
# big burst. The host-side `SerialListTypeDeserializer` must stitch those
# bursts back into a single list. The count field stays 16 bits (byte-aligned,
# header fills the frame), so this exercises the multi-burst *reassembly*
# without depending on the sub-byte frame layout the C++ facade codegen does
# not yet model.
_MULTIBURST_READ_TAG = 0xF00D
_MULTIBURST_READ_ITEMS = [0x1000 + i for i in range(10)]
_MULTIBURST_READ_FIFO_DEPTH = 4


class ChannelMultiBurstListRead(Module):
  """To-host channel that emits a list split across multiple serial bursts.

  Same shape as `ChannelWindowedListRead`, but the ten-element list exceeds
  the serial encoder's data FIFO (depth 4), so `ListWindowToSerial` emits the
  list as several header/data bursts (4 + 4 + 2) followed by a single count==0
  footer. Verifies that the host-side `SerialListTypeDeserializer` reassembles
  a multi-burst transfer back into one list. A write to offset ``0x10`` of the
  ``trigger`` MMIO region arms one transfer.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    # MMIO trigger: any write to offset 0x10 arms one transfer.
    trigger_bundle = esi.MMIO.read_write(appid=AppID("trigger"))
    resp_chan = Wire(Channel(Bits(64)))
    cmd_chan = trigger_bundle.unpack(data=resp_chan)["cmd"]
    cmd_xact, cmd = cmd_chan.snoop_xact()
    resp_chan.assign(cmd_chan.transform(lambda c: Bits(64)(c.data)))
    trigger_xact = cmd_xact & (cmd.offset == UInt(32)(0x10))

    n_items = len(_MULTIBURST_READ_ITEMS)
    burst_end = Wire(Bits(1))

    # ``armed`` is high for the duration of one list: set on a trigger write,
    # cleared on the cycle the final item handshakes. The producer streams all
    # ten items with a single ``last`` on item 9 -- the split into bursts
    # happens entirely inside `ListWindowToSerial`.
    armed = ControlReg(clk, rst, asserts=[trigger_xact], resets=[burst_end])

    par_ready = Wire(Bits(1))
    handshake = armed & par_ready
    idx_counter = Counter(4)(clk=clk,
                             rst=rst,
                             clear=burst_end,
                             increment=handshake,
                             instance_name="multiburst_read_idx")
    idx = idx_counter.out
    last_bit = (idx == UInt(4)(n_items - 1))
    burst_end.assign(handshake & last_bit)
    item_value = Array(Bits(32), n_items)(_MULTIBURST_READ_ITEMS)[idx.as_bits()]

    parallel_window_type = Window.default_of(_window_probe_struct)
    par_struct = parallel_window_type.lowered_type({
        "tag": Bits(16)(_MULTIBURST_READ_TAG),
        "items": item_value,
        "last": last_bit,
    })
    par_window = parallel_window_type.wrap(par_struct)
    parallel_chan, parallel_ready = Channel(parallel_window_type).wrap(
        par_window, armed)
    par_ready.assign(parallel_ready)

    p2s = ListWindowToSerial(parallel_window_type, _WINDOW_PROBE_BULK_WIDTH,
                             _WINDOW_PROBE_ITEMS_PER_FRAME,
                             _MULTIBURST_READ_FIFO_DEPTH)(
                                 clk=clk, rst=rst, parallel_in=parallel_chan)
    esi.ChannelService.to_host(AppID("data"), p2s.serial_out)


class ChannelWindowedListWrite(Module):
  """From-host channel of ``window<{tag, list<si32>}>``.

  Exercises the typed write path for windowed-list-with-header on a raw
  channel. Hardware receives one burst, converts it to parallel, and
  AND-reduces per-beat equality against the same constant pattern as
  `ChannelWindowedListRead`. The latched match flag is exposed via the
  ``match`` MMIO region so the driver can verify the burst landed
  correctly.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    chan = esi.ChannelService.from_host(AppID("data"), _window_probe_window)
    s2p = ListWindowToParallel(_window_probe_window)(clk=clk,
                                                     rst=rst,
                                                     serial_in=chan)
    par_ready = Wire(Bits(1))
    par_window, par_valid = s2p.parallel_out.unwrap(par_ready)
    par_struct = par_window.unwrap()
    par_ready.assign(Bits(1)(1))

    handshake = par_valid
    last_bit = par_struct["last"].as_bits(1)

    # Counter cycles 0..N-1, clears on the burst-end beat.
    n_items = len(_WINDOW_PROBE_ITEMS)
    idx_clr = (handshake & last_bit).as_bits(1)
    idx_counter = Counter(2)(clk=clk,
                             rst=rst,
                             clear=idx_clr,
                             increment=handshake,
                             instance_name="window_write_idx")
    idx = idx_counter.out

    expected_bits = Array(
        Bits(32), len(_WINDOW_PROBE_ITEMS))(_WINDOW_PROBE_ITEMS)[idx.as_bits()]

    tag_ok = (par_struct["tag"].as_bits(16) == Bits(16)(_WINDOW_PROBE_TAG))
    item_ok = (par_struct["items"].as_bits(32) == expected_bits)
    beat_ok = (tag_ok & item_ok).as_bits(1)

    # Running AND-reduce across the burst; latches into ``final_match`` on
    # the burst-end beat.
    running_match = Wire(Bits(1))
    running_match_reg = Reg(Bits(1),
                            clk=clk,
                            rst=rst,
                            rst_value=1,
                            ce=handshake,
                            name="window_match_running")
    running_match.assign((running_match_reg & beat_ok).as_bits(1))
    running_match_reg.assign(running_match)

    final_match = Reg(Bits(1),
                      clk=clk,
                      rst=rst,
                      rst_value=0,
                      ce=idx_clr,
                      name="window_match_final")
    final_match.assign(running_match)

    # Expose the latched flag via MMIO read.
    mmio_bundle = esi.MMIO.read(appid=AppID("match"))
    resp_chan = Wire(Channel(Bits(64)))
    addr_chan = mmio_bundle.unpack(data=resp_chan)["offset"]
    resp_chan.assign(
        addr_chan.transform(
            lambda _: final_match.as_bits(1).pad_or_truncate(64).as_bits(64)))


# Multi-burst bare-list window. The bulk count is 8 bits, so each burst can
# carry at most 255 items; a 256-element list must therefore be split by the
# host serializer into two bursts (255 + 1). Hardware reassembles the bursts
# via `ListWindowToParallel` and validates every item in order, proving the
# write-side burst chunking round-trips through hardware.
#
# The window is a *bare* list of byte-sized items (no static header fields)
# with an 8-bit count, so both the header frame (a single ui8 count) and each
# data frame (one ui8) fill exactly one byte with no frame padding. That keeps
# the on-wire layout unambiguous: a narrower (sub-byte) count would introduce
# sub-byte frame padding, which the C++ facade codegen does not currently
# model.
_MULTIBURST_N_ITEMS = 256
_MULTIBURST_BULK_WIDTH = 8
_MULTIBURST_ITEMS_PER_FRAME = 1
_multiburst_struct = StructType([("items", List(Bits(8)))])
_multiburst_window = Window.serial_of(_multiburst_struct,
                                      _MULTIBURST_BULK_WIDTH,
                                      _MULTIBURST_ITEMS_PER_FRAME)


class ChannelMultiBurstListWrite(Module):
  """From-host channel of ``window<{list<ui8>}>`` with an 8-bit bulk count.

  The 8-bit bulk count caps each burst at 255 items, so the host splits the
  256-element list into two bursts (255 + 1). Hardware reassembles the bursts
  via `ListWindowToParallel`, AND-reduces per-beat equality against the
  expected ``item[i] == i`` pattern, and latches the result into the ``match``
  MMIO region. This is the end-to-end check that the host-side multi-burst
  serialization is correct.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    chan = esi.ChannelService.from_host(AppID("data"), _multiburst_window)
    s2p = ListWindowToParallel(_multiburst_window)(clk=clk,
                                                   rst=rst,
                                                   serial_in=chan)
    par_ready = Wire(Bits(1))
    par_window, par_valid = s2p.parallel_out.unwrap(par_ready)
    par_struct = par_window.unwrap()
    par_ready.assign(Bits(1)(1))

    handshake = par_valid
    last_bit = par_struct["last"].as_bits(1)

    # Counter cycles 0..255, clears on the burst-end beat. Each item's value
    # equals its position, so the expected value is simply the index.
    idx_clr = (handshake & last_bit).as_bits(1)
    idx_counter = Counter(8)(clk=clk,
                             rst=rst,
                             clear=idx_clr,
                             increment=handshake,
                             instance_name="multiburst_write_idx")
    idx = idx_counter.out

    beat_ok = (par_struct["items"].as_bits(8) == idx.as_bits(8)).as_bits(1)

    # Running AND-reduce across the whole (reassembled) list; latches into
    # ``final_match`` on the last item's beat.
    running_match = Wire(Bits(1))
    running_match_reg = Reg(Bits(1),
                            clk=clk,
                            rst=rst,
                            rst_value=1,
                            ce=handshake,
                            name="multiburst_match_running")
    running_match.assign((running_match_reg & beat_ok).as_bits(1))
    running_match_reg.assign(running_match)

    final_match = Reg(Bits(1),
                      clk=clk,
                      rst=rst,
                      rst_value=0,
                      ce=idx_clr,
                      name="multiburst_match_final")
    final_match.assign(running_match)

    # Expose the latched flag via MMIO read.
    mmio_bundle = esi.MMIO.read(appid=AppID("match"))
    resp_chan = Wire(Channel(Bits(64)))
    addr_chan = mmio_bundle.unpack(data=resp_chan)["offset"]
    resp_chan.assign(
        addr_chan.transform(
            lambda _: final_match.as_bits(1).pad_or_truncate(64).as_bits(64)))


class Top(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    TypedFuncMultiArg(clk=ports.clk,
                      rst=ports.rst,
                      appid=AppID("typed_func_multi_arg_inst"))
    TypedFuncVoidArg(clk=ports.clk,
                     rst=ports.rst,
                     appid=AppID("typed_func_void_arg_inst"))
    TypedFuncVoidResult(clk=ports.clk,
                        rst=ports.rst,
                        appid=AppID("typed_func_void_result_inst"))
    CallServiceCallback(clk=ports.clk,
                        rst=ports.rst,
                        appid=AppID("call_service_callback_inst"))
    TypedReadChannelStruct(clk=ports.clk,
                           rst=ports.rst,
                           appid=AppID("typed_read_channel_struct_inst"))
    TypedWriteChannelByte(clk=ports.clk,
                          rst=ports.rst,
                          appid=AppID("typed_write_channel_byte_inst"))
    MmioReadWrite(clk=ports.clk,
                  rst=ports.rst,
                  appid=AppID("mmio_read_write_inst"))
    TelemetryMetric(clk=ports.clk,
                    rst=ports.rst,
                    appid=AppID("telemetry_metric_inst"))
    IndexedFuncGroup(clk=ports.clk,
                     rst=ports.rst,
                     appid=AppID("indexed_func_group_inst"))

    # Two CustomServiceDeclChannel instances at indexed appids exercise the
    # custom-service-decl path AND same-name multi-instance hierarchy.
    CustomServiceDeclChannel(clk=ports.clk,
                             appid=AppID("custom_service_decl_channel", 0))
    CustomServiceDeclChannel(clk=ports.clk,
                             appid=AppID("custom_service_decl_channel", 1))
    TypedFuncStruct(clk=ports.clk,
                    rst=ports.rst,
                    appid=AppID("typed_func_struct_inst"))
    TypedFuncNestedStruct(clk=ports.clk,
                          rst=ports.rst,
                          appid=AppID("typed_func_nested_struct_inst"))
    TypedFuncSubByteSigned(clk=ports.clk,
                           rst=ports.rst,
                           appid=AppID("typed_func_subbyte_signed_inst"))
    TypedFuncArrayResult(clk=ports.clk,
                         rst=ports.rst,
                         appid=AppID("typed_func_array_result_inst"))
    TypedFuncWindowedList(clk=ports.clk,
                          rst=ports.rst,
                          appid=AppID("typed_func_windowed_list_inst"))
    ChannelWindowedListRead(clk=ports.clk,
                            rst=ports.rst,
                            appid=AppID("channel_windowed_list_read_inst"))
    ChannelMultiBurstListRead(clk=ports.clk,
                              rst=ports.rst,
                              appid=AppID("channel_multiburst_list_read_inst"))
    ChannelWindowedListWrite(clk=ports.clk,
                             rst=ports.rst,
                             appid=AppID("channel_windowed_list_write_inst"))
    ChannelMultiBurstListWrite(
        clk=ports.clk,
        rst=ports.rst,
        appid=AppID("channel_multiburst_list_write_inst"))
    CallbackWindowedList(clk=ports.clk,
                         rst=ports.rst,
                         appid=AppID("callback_windowed_list_inst"))


if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2] if len(sys.argv) > 2 else None)
  s = System(bsp(Top), name="TestCodegen", output_directory=sys.argv[1])
  s.compile()
  s.package()
