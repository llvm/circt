# PyCDE development guide

PyCDE is the Python frontend on top of CIRCT. It builds hardware by
constructing MLIR (mostly `hw`, `comb`, `seq`, `esi`, `fsm`) through Python
classes and operator overloads. This document captures conventions and
pitfalls that come up when writing real hardware in PyCDE — particularly
ESI-channel logic — that are not always obvious from the API surface.

## Build and test

PyCDE is built as part of CIRCT. For local agent work, prefer reusing an
existing CIRCT build under `build/default` and a `.venv` at the repo root.
See [lib/Dialect/ESI/runtime/AGENTS.md](../../lib/Dialect/ESI/runtime/AGENTS.md)
for the canonical environment setup; the same exports (`PYTHONPATH`,
`LD_LIBRARY_PATH`, `PATH`) are required for PyCDE work because the runtime
and PyCDE are co-developed.

The PyCDE source under [frontends/PyCDE/src/pycde](src/pycde) is symlinked
into both the build's `python_packages/pycde` and the venv's
`site-packages/pycde`, so edits to `.py` files take effect without
rebuilding. C++ changes (CAPI, dialects) require `ninja -C build/default
PyCDE`.

Common test invocations:

- Lit tests (fast, no simulator):
  `ninja -C build/default check-pycde`
- Integration tests (cosim, requires Verilator):
  `ninja -C build/default check-pycde-integration`
- ESI runtime integration pytests that exercise PyCDE-generated hardware:
  ```bash
  cd lib/Dialect/ESI/runtime
  PATH=$PWD/../../../../build/default/bin:$PATH \
    ../../../../.venv/bin/python -m pytest tests/integration/ -v
  ```

Always run `yapf` on Python changes before committing; CI enforces it.

## Mental model

A PyCDE module is a Python class subclassing `pycde.Module` with `Input`/
`Output`/`Clock`/`Reset` port descriptors and a `@generator` method. The
generator runs **once at elaboration** and constructs the IR. Inside it,
operator overloads on `Signal` subclasses (`BitsSignal`, `UIntSignal`,
`StructSignal`, `ChannelSignal`, ...) emit ops; assignments to `ports.X`
connect outputs.

Key consequence: the generator is plain Python, so loops, dicts, and
helper functions are all allowed and run at elaboration time. There is no
"runtime Python" in the generated hardware — anything that needs to vary
per cycle must be expressed with signals.

## Signals, types, and operators

- Use the typed signal subclasses (`Bits`, `UInt`, `SInt`) rather than
  raw bit manipulation. Width inference is strict: `UInt(N)(x) +
  UInt(N)(y)` is `UInt(N+1)`; cast back with `.as_uint(N)` /
  `.as_bits(N)` when feeding back into a register or comparator.
- `==`, `!=`, `<`, `>`, `&`, `|`, `~`, `^` produce signals. `and`, `or`,
  `not` do **not** — they coerce to Python bool and silently break
  the generator. Always use bitwise operators.
- `Signal.reg(clk, rst, ce=..., rst_value=..., name=...)` is the
  preferred way to create a register; the `name` argument shows up in
  the generated Verilog and waveform dumps. Use it liberally — debugging
  `_T_42` is painful.
- Struct field access: `s["field"]` (string keying) is the portable form;
  `s.field` works only when there are no name collisions with Python
  attributes.
- Zero-width signals are illegal in `comb` (notably `comb.icmp`) and
  surface as MLIR legalization failures rather than Python errors.
  When defaulting parameters like `meta_fifo_depth=fifo_depth // 4`,
  clamp with `max(2, ...)` to keep widths positive.

## Forward declarations: `Wire`

Use `Wire(Bits(N))` (or any type) when a signal is needed before it can
be defined — typically because the FSM next-state, ready, or counter
expressions are mutually recursive. Drive it exactly once via
`wire.assign(expr)`. Forgetting the `assign` is a silent
elaboration-time bug; double assignment raises.

## Multiplexers and `ControlReg`

- `Mux(sel, a, b, ...)` from `pycde.constructs` accepts a `clog2(N)`-bit
  selector indexing N inputs (or a 1-bit selector for the 2-input form).
  Prefer a multi-input `Mux(state_wire, in_s0, in_s1, in_s2, in_s3)` over
  cascaded `Mux(a, Mux(b, ...))` when the selection is naturally
  state-indexed — it reads better and lowers to a flat case.
- `ControlReg(clk, rst, asserts=[...], resets=[...], name=...)` returns a
  `BitsSignal(1)` that stays high after any `assert` pulse and goes low
  after any `reset` pulse. `asserts` has priority over `resets` on the
  same cycle. This is the right primitive for "set on one event, clear
  on another" flags (e.g. `in_list`, `busy`); it is much clearer than a
  hand-rolled `(prev | set) & ~clear` register and avoids reset/CE
  ordering mistakes.

## Counters

`pycde.constructs.Counter` increments on `increment` and clears on
`clear`; **clear takes precedence over increment**. Use it instead of
hand-rolled `(reg + 1).reg(ce=...)` patterns when there is a natural
"reset to zero on event X" condition; it eliminates a class of
off-by-one bugs around the boundary cycle.

## ESI channels and handshake

The most common subtle bugs in PyCDE involve valid/ready handshake
mistakes on ESI channels.

### Unwrap / wrap

```python
data, valid = ports.in_chan.unwrap(ready_wire)
out_chan, ready = Channel(T).wrap(data, valid)
ports.out_chan = out_chan
```

A "transaction" on a channel happens **only** when both `valid` and
`ready` are high. Driving `ready` from anything that itself depends on
`valid` from the same direction creates a combinational cycle; driving
state transitions from `valid` alone double-counts beats whenever the
consumer back-pressures.

### The two cardinal rules

1. **An FSM event that consumes a beat must be qualified by `valid &
   ready`, not by `valid` alone.** If the producer is presenting a beat
   but downstream is not ready, `valid` stays high for many cycles —
   re-latching headers or advancing counters on bare `valid` will
   corrupt state.

2. **Do not gate upstream `ready` on a downstream `ready` that may itself
   be a function of our `valid` if our `valid` is zero in that state.**
   The ESI spec permits ready-as-a-function-of-valid downstreams, and
   this combination deadlocks: we wait for `out_ready`, downstream waits
   for `out_valid`, neither moves. The fix is a per-state ready policy:
   in states where we are not driving an output beat (e.g. consuming a
   header into a register, latching silently into a buffer), assert
   upstream `ready` unconditionally; only gate by `out_ready` in states
   where the consumed beat is being forwarded.

The repo convention for the simple forwarding case (and what
`Channel.join` / `ChannelSignal.fork` do internally) is:

```
upstream.ready = downstream.ready & xact_predicate
```

where `xact_predicate` includes `valid` of the channels actually being
forwarded. See [src/pycde/types.py](src/pycde/types.py) and
[src/pycde/signals.py](src/pycde/signals.py) for the canonical patterns.

## FSM idioms

For small, regular state machines, prefer to encode the state as a
narrow `Bits` enum and drive next-state via `Mux(state_wire, next_s0,
next_s1, ...)`. Use the `pycde.fsm` dialect helper for larger machines
where transition tables and entry/exit actions become unwieldy.

When the FSM controls a handshake, derive **all** of the following from
the same canonical `state_wire`:

- per-state `in_X` predicates (e.g. `in_wait = state_wire == S_WAIT`)
- the next-state `Mux`
- the per-state `ready` policy
- the per-state `valid` of any output channel
- any "transaction occurred" predicates used to advance counters or
  latch registers

This makes the per-state contract reviewable in one place and avoids the
"some signals were updated, others weren't" drift that produces
deadlocks and dropped beats.

## Telemetry and debugging generated hardware

- Use `Counter` + a `MMIO`/`AppID`-tagged read register to surface
  invariant violations (e.g. "static field changed mid-list") to host software.
  Cheap on area and invaluable when a cosim test fails far from the bug.
  Alternatively, 'esi.Telemetry' provides a more structured interface for this
  pattern. It can be automatically read via the esiaccel API and the esiquery
  tool.
- Prefer `.reg(name="...")` everywhere; the names propagate to Verilog
  and to waveform viewers.
- For non-trivial generators, write a small lit test under [test/](test/) that
  runs a module decorated with `unittest()` and FileChecks the IR, in addition
  to any cosim integration test. Lit tests are seconds; cosim tests are minutes.

## Style

- Match the existing style in `src/pycde/`: snake_case for locals and
  signal names, PascalCase for module classes and types,
  `S_UPPER_SNAKE` for state constants.
- Top-level docstrings on `Module` subclasses describe the protocol
  contract (when ports are valid, what is preserved across resets,
  etc.). Do not skimp on these — the hardware contract is harder to
  recover from the code than a software contract.
- Run `yapf` before committing. CI runs `yapf --diff` and rejects
  unformatted changes.
