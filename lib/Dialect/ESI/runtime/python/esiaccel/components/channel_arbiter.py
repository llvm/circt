# ===- channel_arbiter.py - pipelined list-aware channel mux -------------===//
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//
#
#  A high-performance, pipelined, list-aware N:1 ESI channel multiplexer. See
#  `docs/components/ChannelArbiter.md` for the design details.
#
# ===----------------------------------------------------------------------===//

from typing import List, Optional, Tuple

from pycde import AppID, Clock, Input, Module, Output, Reset, generator
from pycde.constructs import Counter, Mux, Reg, Wire
from pycde.esi import Telemetry
from pycde.module import modparams
from pycde.seq import FIFO as SeqFIFO
from pycde.signals import (BitsSignal, ChannelSignal, ClockSignal, Or, Signal)
from pycde.support import clog2
from pycde.types import (Array, Bits, Channel, ChannelSignaling, StructType,
                         UInt, Window)


def _select_reg_levels(num_inputs: int,
                       mux_pipeline_levels: Optional[int]) -> List[int]:
  """Tree levels after which `_select_mux` inserts a pipeline register.

  A register is placed after every `mux_pipeline_levels` levels, except after
  the final (root) level -- its result is registered downstream. This is the
  single source of truth for the mux-tree pipelining: `_select_mux` builds the
  registers at these levels and `_select_latency` just counts them."""
  if num_inputs <= 1 or not mux_pipeline_levels:
    return []
  gw = clog2(num_inputs)
  return [
      level for level in range(gw)
      if (level + 1) % mux_pipeline_levels == 0 and level < gw - 1
  ]


def _select_latency(num_inputs: int, mux_pipeline_levels: Optional[int]) -> int:
  """Pipeline-register latency (cycles) that `_select_mux` inserts."""
  return len(_select_reg_levels(num_inputs, mux_pipeline_levels))


def _select_mux(sel: BitsSignal, values: List[BitsSignal], clk: ClockSignal,
                rst: Signal, mux_pipeline_levels: Optional[int]) -> BitsSignal:
  """Return `values[sel]`.

  With `mux_pipeline_levels` falsy this is a flat combinational mux (a single
  `hw.array_get`, which CIRCT lowers to an unpipelined mux tree). Otherwise it
  is built as an explicit balanced binary mux tree -- 2:1 nodes consuming one
  `sel` bit per level -- with a pipeline register inserted after every
  `mux_pipeline_levels` levels. This lets a large/wide selection mux (the
  Fmax bottleneck of a big fan-in mux) be retimed across registers. The
  remaining `sel` bits are pipelined alongside the partial results so each
  level selects with the correctly-delayed index. The added latency is
  `_select_latency(len(values), mux_pipeline_levels)` cycles."""
  n = len(values)
  if n == 1:
    return values[0]
  if not mux_pipeline_levels:
    return Mux(sel, *values)
  gw = clog2(n)
  reg_levels = set(_select_reg_levels(n, mux_pipeline_levels))
  # Pad to a full 2**gw-leaf tree; padded leaves carry a never-selected copy
  # (the index is always < n).
  cur = list(values) + [values[0]] * ((1 << gw) - n)
  rem = sel
  for level in range(gw):
    bit = rem[0]
    cur = [Mux(bit, cur[2 * i], cur[2 * i + 1]) for i in range(len(cur) // 2)]
    if rem.type.width > 1:
      rem = rem[1:]
    if level in reg_levels:
      cur = [c.reg(clk, rst) for c in cur]
      rem = rem.reg(clk, rst)
  return cur[0]


def _onehot_to_index(onehot: BitsSignal) -> BitsSignal:
  """Encode a one-hot bit-vector to its binary index. Bit `b` of the result is
  the OR of the one-hot bits whose index has bit `b` set."""
  num_inputs = onehot.type.width
  bits = []
  for b in range(clog2(num_inputs)):
    terms = [onehot[i] for i in range(num_inputs) if (i >> b) & 1]
    bits.append(Or(*terms) if terms else Bits(1)(0))
  return BitsSignal.concat(list(reversed(bits)))


# Grant-control strategies. `GrantSchedulerMod` and `RoundRobinControlMod` are
# interchangeable: they deliberately carry the *same* port signature, documented
# per-port on `GrantScheduler` below, so the arbiter picks one and wires it up
# identically. (PyCDE scans only a class's own dict for ports, so the signature
# cannot be inherited from a common base -- it is spelled out in each and must be
# kept in sync.) `launch` is unused by the round-robin strategy; it is present so
# the signature stays uniform.
#
# A control module owns the grant FSM state (`grant`/`grant_oh`/`busy`, plus
# whatever else the strategy needs) and exposes it for the datapath to read. Its
# inputs are purely observations of the datapath: which inputs are offering
# (`valids`), and whether a flit / a final flit was accepted (`launch`,
# `msg_end`). `_build_grant_state` below builds the state common to both.


def _build_grant_state(
    ports, clk: ClockSignal, rst: Signal, num_inputs: int,
    next_grant: BitsSignal,
    next_busy: BitsSignal) -> Tuple[BitsSignal, BitsSignal, BitsSignal]:
  """Register `next_grant`/`next_busy` into the grant FSM state every control
  module has, drive the `grant`/`grant_oh`/`busy` ports with it, and return
  `(grant, grant_oh, busy)` for the strategy to compute its next state from
  (typically via `Wire`s, since next state depends on current).

  `grant_oh` is decoded *ahead* of its registers -- one flop per input -- so
  each high-fanout per-input grant is driven straight from a flop rather than a
  shared combinational decode of `grant`. Decoding at the instantiation site
  would necessarily land after the register, hence it lives here. Both are fed
  from the same next-state, so `grant_oh[i]` is high exactly when
  `grant == i`."""
  gw = clog2(num_inputs)
  grant = next_grant.reg(clk, rst, name="grant")
  grant_oh = BitsSignal.concat([
      (next_grant == Bits(gw)(i)).reg(clk,
                                      rst,
                                      rst_value=(1 if i == 0 else 0),
                                      name=f"grant_oh_{i}")
      for i in reversed(range(num_inputs))
  ])
  busy = next_busy.reg(clk, rst, name="busy")
  ports.grant = grant
  ports.grant_oh = grant_oh
  ports.busy = busy
  return grant, grant_oh, busy


@modparams
def GrantSchedulerMod(num_inputs: int, queue_depth: int):
  """Decoupled, pipelinable grant scheduler (`pipelined_scheduler=True`).

  A **grant queue** holds upcoming winners for the datapath to pop, and a
  **sweep scheduler** refills it off the critical path. That breaks the flat
  arbiter's single-cycle `grant -> grant` loop, its dominant timing limiter at
  high fan-in. A queued entry is a hint about who to serve next, not a promise
  that a particular message is waiting: an entry whose input has since gone
  idle is skipped in one cycle (`stale` below) rather than stalling the output.

  Consequently service order is best-effort, and `queue_depth` bounds only how
  far ahead of the datapath decisions are committed -- it is not a fairness
  knob. See section 7.1 of `docs/components/ChannelArbiter.md` for why
  committing early is safe and for the full ordering/latency caveats."""
  assert num_inputs >= 2, "GrantSchedulerMod requires at least two inputs"
  gw = clog2(num_inputs)

  class GrantScheduler(Module):
    clk = Clock()
    rst = Reset()

    # Per-input `valid`; bit `i` is high when input `i` is offering a flit.
    valids = Input(Bits(num_inputs))
    # High on cycles a flit is accepted from the granted input.
    launch = Input(Bits(1))
    # High on the `launch` of a message's final flit.
    msg_end = Input(Bits(1))

    # Index of the currently granted input.
    grant = Output(Bits(gw))
    # `grant` pre-decoded to one-hot, one register per bit.
    grant_oh = Output(Bits(num_inputs))
    # High while `grant` is in force, i.e. an input is currently being served.
    busy = Output(Bits(1))
    # High on cycles the grant is (re)loaded from the queue; telemetry only.
    switch = Output(Bits(1))

    @generator
    def build(ports) -> None:
      clk = ports.clk
      rst = ports.rst
      next_grant = Wire(Bits(gw), "next_grant")
      next_busy = Wire(Bits(1), "next_busy")
      grant, grant_oh, busy = _build_grant_state(ports, clk, rst, num_inputs,
                                                 next_grant, next_busy)

      # Grant queue. `rd_latency=0` makes it show-ahead, so `q_head` is a
      # registered value available the same cycle -- popping adds no bubble.
      gq = SeqFIFO(Bits(gw), queue_depth, clk, rst)
      q_pop = Wire(Bits(1), "gq_pop")
      q_head = gq.pop(q_pop)
      q_nonempty = ~gq.empty

      # ---- Sweep scheduler (off the datapath's critical path). ----
      pending = Reg(Bits(num_inputs),
                    clk,
                    rst,
                    rst_value=0,
                    name="sched_pending")
      pend_nonzero = pending != Bits(num_inputs)(0)
      # Isolate the lowest set bit: x & (-x), with -x == ~x + 1.
      neg_pending = ((~pending).as_uint(num_inputs) +
                     UInt(num_inputs)(1)).as_bits(num_inputs)
      low = pending & neg_pending
      push = pend_nonzero & ~gq.full
      gq.push(_onehot_to_index(low), push)

      # Clear the bit just scheduled (or hold if the queue is full), and reload
      # the snapshot as soon as the sweep is exhausted. The reload has to happen
      # on the *same* cycle the last bit is pushed: deferring it to the cycle
      # after `pending` reads zero costs one idle cycle per sweep, capping
      # throughput at `n/(n+1)` for `n` concurrently-active inputs. That only
      # bites for single-flit messages; with multi-flit lists the datapath is
      # still streaming the current message while the sweep refills, so the
      # bubble is hidden.
      cleared = pending & ~low
      sweep_done = push & (cleared == Bits(num_inputs)(0))
      pending.assign(
          Mux(~pend_nonzero | sweep_done, Mux(push, pending, cleared),
              ports.valids))

      # ---- Datapath grant FSM. ----
      # `started` distinguishes "this grant has not delivered a flit yet" (safe
      # to abandon) from "mid-message" (abandoning would split the message).
      started = Reg(Bits(1), clk, rst, rst_value=0, name="grant_started")
      sel_valid_now = (ports.valids & grant_oh).or_reduce()
      # Abandon a grant that has not yet delivered a flit and whose input is not
      # offering one, but only when there is someone else to serve.
      stale = busy & ~started & ~sel_valid_now & q_nonempty
      advance = ports.msg_end | stale
      take_next = ~busy | advance
      q_pop.assign(take_next & q_nonempty)

      next_grant.assign(Mux(take_next & q_nonempty, grant, q_head))
      next_busy.assign(Mux(take_next, busy, q_nonempty))
      # Taking a new grant clears `started`; otherwise the first launched flit
      # sets it.
      started_next = Mux(ports.launch, started, Bits(1)(1))
      started.assign(Mux(take_next, started_next, Bits(1)(0)))

      # The grant is replaced by a queued decision exactly when it is popped.
      ports.switch = q_pop

  return GrantScheduler


@modparams
def RoundRobinArbiterMod(num_inputs: int):
  """Combinational round-robin winner selection, factored into its own module
  for waveform visibility.

  Given a per-input `valids` bitmask (bit `i` is input `i`) and a `start` index,
  `winner` is the lowest-index input that is valid and at index `>= start`
  (cyclically), falling back to the lowest-index valid input overall; `any_valid`
  is high when any input is valid. Purely combinational -- the owning state
  (`rr_ptr`, `grant`/`busy`) lives in `RoundRobinControlMod`."""
  assert num_inputs >= 2, "RoundRobinArbiterMod requires at least two inputs"
  gw = clog2(num_inputs)

  class RoundRobinArbiter(Module):
    valids = Input(Bits(num_inputs))
    start = Input(Bits(gw))
    winner = Output(Bits(gw))
    any_valid = Output(Bits(1))

    @generator
    def build(ports) -> None:

      def priority_lsb(
          bits_list: List[BitsSignal]) -> Tuple[BitsSignal, BitsSignal]:
        """Index of the lowest-index set bit, plus an any-set flag, computed as
        a balanced binary tree (O(log N) depth) rather than an O(N) chain. Each
        node combines two subtrees, giving priority to the lower index, and
        prefixes the selected sub-index with the branch bit."""
        # Leaves carry (any, sub-index); pad up to 2**gw with never-set leaves
        # so the tree is perfect and each level consumes one index bit.
        level = [(b, None) for b in bits_list]
        level += [(Bits(1)(0), None) for _ in range((1 << gw) - len(bits_list))]
        width = 0
        while len(level) > 1:
          nxt = []
          for j in range(0, len(level), 2):
            la, li = level[j]
            ra, ri = level[j + 1]
            # The lower-index (left) subtree wins if it has any set bit.
            take_right = ~la
            if width == 0:
              idx = take_right
            else:
              idx = BitsSignal.concat([take_right, Mux(take_right, li, ri)])
            nxt.append((la | ra, idx))
          level = nxt
          width += 1
        idx = level[0][1]
        return (idx if idx is not None else Bits(gw)(0)), level[0][0]

      valid_bits = [ports.valids[i] for i in range(num_inputs)]
      start_u = ports.start.as_uint(gw)
      # Winner among inputs at-or-after `start`, else the lowest-index winner.
      hi = [valid_bits[i] & (UInt(gw)(i) >= start_u) for i in range(num_inputs)]
      hi_idx, hi_any = priority_lsb(hi)
      lo_idx, lo_any = priority_lsb(valid_bits)
      ports.winner = Mux(hi_any, lo_idx, hi_idx)
      ports.any_valid = hi_any | lo_any

  return RoundRobinArbiter


@modparams
def RoundRobinControlMod(num_inputs: int):
  """Flat round-robin grant control (the default strategy).

  Answers "who is granted next?" combinationally in the cycle the current
  message ends, using two `RoundRobinArbiter` instances -- one for picking up
  from idle, one for the message-end turnaround -- plus the `rr_ptr` fairness
  pointer, which is private to this strategy. See section 7 of
  `docs/components/ChannelArbiter.md`.

  `launch` is unused; it exists only to match `GrantSchedulerMod`'s
  signature."""
  assert num_inputs >= 2, "RoundRobinControlMod requires at least two inputs"
  gw = clog2(num_inputs)

  class RoundRobinControl(Module):
    clk = Clock()
    rst = Reset()

    valids = Input(Bits(num_inputs))
    launch = Input(Bits(1))
    msg_end = Input(Bits(1))

    grant = Output(Bits(gw))
    grant_oh = Output(Bits(num_inputs))
    busy = Output(Bits(1))
    switch = Output(Bits(1))

    @generator
    def build(ports) -> None:
      clk = ports.clk
      rst = ports.rst
      next_grant = Wire(Bits(gw), "next_grant")
      next_busy = Wire(Bits(1), "next_busy")
      grant, grant_oh, busy = _build_grant_state(ports, clk, rst, num_inputs,
                                                 next_grant, next_busy)
      rr_ptr = Reg(Bits(gw), clk, rst, name="rr_ptr")
      rr_arbiter = RoundRobinArbiterMod(num_inputs)

      def round_robin(valids_vec: BitsSignal, start: BitsSignal,
                      name: str) -> Tuple[BitsSignal, BitsSignal]:
        """Instantiate a RoundRobinArbiter over `valids_vec` starting from
        `start`."""
        inst = rr_arbiter(valids=valids_vec, start=start, instance_name=name)
        return inst.winner, inst.any_valid

      grant_u = grant.as_uint(gw)
      is_last_idx = grant == Bits(gw)(num_inputs - 1)
      grant_p1 = Mux(is_last_idx, (grant_u + UInt(gw)(1)).as_bits(gw),
                     Bits(gw)(0))

      winner_idle, any_idle = round_robin(ports.valids, rr_ptr, "rr_idle")
      # At a message end the just-consumed input still asserts `valid` this
      # cycle (the flit is consumed on the clock edge), so mask it out of the
      # re-arbitration. Otherwise the round-robin wrap-around would
      # speculatively re-grant that stale valid and the FSM would get stuck
      # `busy` on an input that goes empty next cycle. A genuinely backlogged
      # input is re-selected on the following idle cycle instead.
      valids_next = ports.valids & ~grant_oh
      winner_next, any_next = round_robin(valids_next, grant_p1, "rr_next")

      pick = ~busy & any_idle
      reend = busy & ports.msg_end
      grant_if_not_reend = Mux(pick, grant, winner_idle)
      busy_if_not_reend = Mux(pick, busy, Bits(1)(1))

      next_grant.assign(Mux(reend, grant_if_not_reend, winner_next))
      next_busy.assign(Mux(reend, busy_if_not_reend, any_next))
      rr_ptr.assign(Mux(reend, rr_ptr, grant_p1))
      ports.switch = pick | (reend & any_next)

  return RoundRobinControl


@modparams
def ChannelArbiterMod(channel_type: Channel, num_inputs: int,
                      output_fifo_depth: int, buffer_inputs: bool,
                      telemetry: bool, mux_pipeline_levels: Optional[int],
                      pipelined_scheduler: bool, grant_queue_depth: int):
  """Build a pipelined, list-aware N:1 channel multiplexer module. See the
  `ChannelArbiter` convenience function for the user-facing entry point and
  `docs/components/ChannelArbiter.md` for the design."""

  assert num_inputs >= 2, "ChannelArbiterMod requires at least two inputs"
  inner = channel_type.inner_type

  # Determine the bit width of the datapath and whether the payload is a list
  # window (which carries a per-flit 'last' field).
  is_window = isinstance(inner, Window)
  if is_window:
    lowered = inner.lowered_type
    field_names = [n for n, _ in lowered.fields] if isinstance(
        lowered, StructType) else None
    if field_names is None or "last" not in field_names:
      raise TypeError(
          "ChannelArbiter can only auto-detect list framing for window types "
          "whose lowered frame is a struct with a 'last' field; got lowered "
          f"type {lowered}. (Serial/union-framed windows are not supported.)")
    width = lowered.bitwidth
  else:
    width = inner.bitwidth
  if width is None:
    raise TypeError(
        f"ChannelArbiter requires a fixed-width payload; got {inner}")

  # The FIFO beat is just the raw payload bits (for list/window payloads the
  # per-flit 'last' flag is already part of them). A zero-width (token) payload
  # carries no data, so it has no beat/FIFO at all -- the output stage uses an
  # outstanding-beat counter instead (SeqFIFO also requires a non-zero width).
  beat_type = Bits(width)

  # Input-index width. (The credit-counter width depends on the resolved
  # output-FIFO depth and is computed in the generator.)
  gw = clog2(num_inputs)

  # Latency (cycles) added when the selection mux is pipelined into a tree.
  tree_latency = (0 if width == 0 else _select_latency(num_inputs,
                                                       mux_pipeline_levels))
  # One register latches the mux result before the FIFO, so the total
  # launch-to-FIFO pipeline latency is the mux-tree latency plus one.
  pipe_latency = tree_latency + 1
  if output_fifo_depth is not None and \
      output_fifo_depth <= pipe_latency:
    raise ValueError(
        f"output_fifo_depth ({output_fifo_depth}) must be > the pipeline "
        f"latency ({pipe_latency})")

  class ChannelArbiterImpl(Module):
    # Extra output-FIFO depth over the pipeline length, covering the credit
    # round-trip; private and class-scoped.
    _SLACK = 2

    clk = Clock()
    rst = Reset()

    inputs = Input(Array(channel_type, num_inputs))
    output = Output(channel_type)

    @generator
    def build(ports) -> None:
      # Resolve the output-FIFO depth (defaulting from the private,
      # class-scoped `_SLACK`, and covering the pipeline latency) and the
      # credit-counter width.
      depth = (pipe_latency + ChannelArbiterImpl._SLACK
               if output_fifo_depth is None else output_fifo_depth)
      cw = max(1, depth.bit_length())
      clk = ports.clk
      rst = ports.rst

      def flit_last(typed_sig: Signal) -> BitsSignal:
        """High when 'typed_sig' is the last flit of its message."""
        if is_window:
          return typed_sig.unwrap()["last"]
        return Bits(1)(1)

      def to_bits(typed_sig: Signal) -> BitsSignal:
        """Bitcast the payload to raw bits for the datapath."""
        if is_window:
          typed_sig = typed_sig.unwrap()
        return typed_sig.bitcast(Bits(width))

      def from_bits(bits: BitsSignal) -> Signal:
        """Reconstruct the payload from raw bits for the output channel."""
        if is_window:
          return inner.wrap(bits.bitcast(inner.lowered_type))
        return bits.bitcast(inner)

      # ---- Arbiter state. `grant`/`grant_oh`/`busy` are owned and registered
      # by the grant-control module instantiated below; these wires forward-
      # declare them because the input stage reads them first. ----
      grant = Wire(Bits(gw), "grant")
      grant_oh = Wire(Bits(num_inputs), "grant_oh")
      busy = Wire(Bits(1), "busy")
      credit = Reg(UInt(cw), clk, rst, rst_value=depth, name="credit")

      credit_gt0 = credit > UInt(cw)(0)

      # ---- Inputs: optional skid buffer, then unwrap with a local ready. ----
      valids: List[BitsSignal] = []
      last_bits: List[BitsSignal] = []
      data_bits: List[BitsSignal] = []
      for i in range(num_inputs):
        chan = ports.inputs[i]
        if buffer_inputs:
          chan = chan.buffer(clk, rst, stages=1)
        # ready[i]: consume only the granted input, and only when a credit is
        # available. Independent of valid, so no combinational ready loop.
        ready_i = busy & grant_oh[i] & credit_gt0
        data_i, valid_i = chan.unwrap(ready_i)
        valids.append(valid_i)
        last_bits.append(flit_last(data_i))
        data_bits.append(to_bits(data_i))

      # ---- Select the granted input. ----
      sel_valid = Mux(grant, *valids)
      sel_last = Mux(grant, *last_bits)
      if width == 0:
        sel_bits = Bits(0)(0)
      else:
        sel_bits = _select_mux(grant, data_bits, clk, rst, mux_pipeline_levels)

      # A beat is launched into the pipeline when the granted input is valid and
      # a credit is available.
      launch = busy & sel_valid & credit_gt0
      msg_end = launch & sel_last

      # ---- Output stage (feed-forward, no backpressure). ----
      # `pop` returns to the arbiter only through the registered credit counter,
      # so the datapath never stalls. The zero-width case is the datapath case
      # minus the data: no pipeline and no FIFO -- the credit counter itself is
      # the token buffer, and a token is available whenever one is in flight.
      if width == 0:
        out_valid = credit < UInt(cw)(depth)  # in-flight (depth - credit) > 0
        payload_bits = Bits(0)(0)
        fifo_pop = None
      else:
        # Delay the launch/valid to match the mux-tree pipeline, add one output
        # register, then buffer the beat in the FIFO.
        pipe_valid = launch
        for _ in range(tree_latency):
          pipe_valid = pipe_valid.reg(clk, rst)
        pipe_valid = pipe_valid.reg(clk, rst, name="pipe_valid")
        pipe_beat = sel_bits.reg(clk, rst, name="pipe_beat")
        fifo = SeqFIFO(beat_type, depth, clk, rst)
        fifo.push(pipe_beat, pipe_valid)
        fifo_pop = Wire(Bits(1), "arb_pop")
        out_valid = ~fifo.empty
        payload_bits = fifo.pop(fifo_pop)

      out_chan, out_ready = channel_type.wrap(from_bits(payload_bits),
                                              out_valid)
      ports.output = out_chan
      pop = out_valid & out_ready
      if fifo_pop is not None:
        fifo_pop.assign(pop)

      # ---- Credit accounting: credit = depth - in-flight. ----
      next_credit = ((credit + pop.as_uint(cw)).as_uint(cw) -
                     launch.as_uint(cw)).as_uint(cw)
      credit.assign(next_credit)

      # ---- Arbitration. ----
      # Either grant-control strategy presents the same ports, so the only
      # difference here is which module gets instantiated.
      ctrl_mod = (GrantSchedulerMod(num_inputs, grant_queue_depth)
                  if pipelined_scheduler else RoundRobinControlMod(num_inputs))
      ctrl = ctrl_mod(clk=clk,
                      rst=rst,
                      valids=BitsSignal.concat(list(reversed(valids))),
                      launch=launch,
                      msg_end=msg_end,
                      instance_name="arb_ctrl")
      grant.assign(ctrl.grant)
      grant_oh.assign(ctrl.grant_oh)
      busy.assign(ctrl.busy)
      arb_switch = ctrl.switch

      # ---- Telemetry. ----
      if telemetry:
        Telemetry.report_signal(clk, rst, AppID("selectedChannel"), grant)
        Telemetry.report_signal(clk, rst, AppID("busy"), busy)

        for i in range(num_inputs):
          served = Counter(64)(clk=clk,
                               rst=rst,
                               clear=Bits(1)(0),
                               increment=launch & grant_oh[i])
          Telemetry.report_signal(clk, rst, AppID(f"grantCount_{i}"),
                                  served.out)

        total_flits = Counter(64)(clk=clk,
                                  rst=rst,
                                  clear=Bits(1)(0),
                                  increment=launch)
        Telemetry.report_signal(clk, rst, AppID("totalFlits"), total_flits.out)
        total_msgs = Counter(64)(clk=clk,
                                 rst=rst,
                                 clear=Bits(1)(0),
                                 increment=msg_end)
        Telemetry.report_signal(clk, rst, AppID("totalMessages"),
                                total_msgs.out)
        arb_switches = Counter(64)(clk=clk,
                                   rst=rst,
                                   clear=Bits(1)(0),
                                   increment=arb_switch)
        Telemetry.report_signal(clk, rst, AppID("arbSwitches"),
                                arb_switches.out)

        # Max per-message flit count.
        cur_len = Counter(32)(clk=clk, rst=rst, clear=msg_end, increment=launch)
        msg_len = (cur_len.out + UInt(32)(1)).as_uint(32)
        max_len = Reg(UInt(32), clk, rst, rst_value=0, name="max_list_len")
        is_new_max = msg_end & (msg_len > max_len)
        max_len.assign(Mux(is_new_max, max_len, msg_len))
        Telemetry.report_signal(clk, rst, AppID("maxListLen"), max_len)

        # Max output in-flight occupancy (depth - credit).
        occ = (UInt(cw)(depth) - credit).as_uint(cw)
        inflight_hw = Reg(UInt(cw), clk, rst, rst_value=0, name="inflight_hw")
        is_new_hw = occ > inflight_hw
        inflight_hw.assign(Mux(is_new_hw, inflight_hw, occ))
        Telemetry.report_signal(clk, rst, AppID("inflightHighWater"),
                                inflight_hw)

  return ChannelArbiterImpl


def ChannelArbiter(input_channels: List[ChannelSignal],
                   clk: ClockSignal,
                   rst: Signal,
                   *,
                   appid: Optional[AppID] = None,
                   output_fifo_depth: Optional[int] = None,
                   buffer_inputs: bool = True,
                   mux_pipeline_levels: Optional[int] = None,
                   pipelined_scheduler: bool = False,
                   grant_queue_depth: int = 4,
                   telemetry: bool = True) -> ChannelSignal:
  """Build a pipelined, list-aware N:1 channel multiplexer.

  Unlike the combinational `pycde.esi.ChannelMux`, this is a flat registered
  round-robin arbiter with a feed-forward output stage (output register + FIFO
  + credit counter), so it closes timing at high fan-in. It also keeps
  multi-flit list messages contiguous: once an input is granted, it holds the
  output until a flit whose 'last' field is set has been transferred. List
  framing is auto-detected from the channel type (window payloads with a 'last'
  field); all other payloads are treated as single-flit messages.

  Arguments:
    input_channels: the channels to multiplex. All must share the same
      (ValidReady) type.
    clk, rst: clock and reset.
    appid: optional `AppID` for the arbiter instance (e.g. to address it or to
      disambiguate its telemetry in the appid hierarchy).
    output_fifo_depth: depth of the output FIFO; must be greater than the
      pipeline latency (one output register plus any selection-mux pipeline
      latency). Defaults to that plus a small internal slack.
    buffer_inputs: insert a per-input skid buffer to localize backpressure.
    mux_pipeline_levels: if set, build the N:1 data-selection mux as an explicit
      binary tree and insert a pipeline register after every this-many tree
      levels (1 = register every level). This retimes the wide selection mux
      for very large fan-in; the added latency is absorbed by the output FIFO /
      credit counter. `None` (default) uses a flat combinational mux.
    pipelined_scheduler: decouple grant selection from the datapath using a
      grant queue fed by a sweep scheduler, instead of re-arbitrating
      combinationally at each message end. This takes the round-robin tree out
      of the single-cycle `grant -> grant` loop, which is the Fmax limiter at
      high fan-in. Changes the service order (see `GrantSchedulerMod`).
    grant_queue_depth: depth of that grant queue -- how many grant decisions
      may be committed ahead of the datapath. Must be >= 2: a single entry
      cannot keep the datapath fed back to back, so every message would cost a
      refill bubble. This is not a fairness knob; a newly-valid input's wait
      also scales with the number of concurrently active inputs (see
      `GrantSchedulerMod`).
    telemetry: emit telemetry (selected channel, list-length stats, etc.).

  See `docs/components/ChannelArbiter.md`."""

  assert len(input_channels) > 0
  num_inputs = len(input_channels)
  if num_inputs == 1:
    return input_channels[0]

  channel_type = input_channels[0].type
  for c in input_channels:
    if c.type != channel_type:
      raise TypeError("All ChannelArbiter inputs must have the same type; got "
                      f"{channel_type} and {c.type}")
  if channel_type.signaling != ChannelSignaling.ValidReady:
    raise TypeError("ChannelArbiter requires ValidReady channels; got "
                    f"{channel_type}")

  if mux_pipeline_levels is not None and mux_pipeline_levels < 1:
    raise ValueError(
        f"mux_pipeline_levels must be >= 1, got {mux_pipeline_levels}")

  # Validated here rather than left to the FIFO: a bad depth otherwise surfaces
  # as a `seq.fifo` verifier error from deep inside the lowering, with no
  # mention of the knob that caused it. Depth 1 is rejected too -- `push` is
  # blocked whenever the queue is non-empty, so a single entry can never keep
  # the datapath fed back to back and every message would cost a refill
  # bubble, silently undoing the Fmax win the option exists for.
  if pipelined_scheduler and grant_queue_depth < 2:
    raise ValueError(f"grant_queue_depth must be >= 2, got {grant_queue_depth}")

  mod = ChannelArbiterMod(channel_type, num_inputs, output_fifo_depth,
                          buffer_inputs, telemetry, mux_pipeline_levels,
                          pipelined_scheduler, grant_queue_depth)
  inputs_array = Array(channel_type, num_inputs)(input_channels)
  inst = mod(clk=clk, rst=rst, inputs=inputs_array, appid=appid)
  return inst.output
