# Probe Dialect

The Probe dialect provides SSA handles for observing hardware values. A probe
handle can be passed through module outputs and resolved where the observed
value is needed, without representing that connection as an ordinary hardware
value or committing to a particular hierarchical path representation.

[TOC]

## Rationale

Hardware IR often needs to preserve an observation relationship across module
boundaries. Routing the observed value through ordinary ports changes the
module interface and may introduce unnecessary data dependencies. The Probe
dialect represents this relationship explicitly: `!probe.ref<T>` is a
read-only handle to a value of type `T`, created by `probe.send` and observed
by `probe.read`.

A probe reference is not a hardware value and cannot be driven. Its payload
must be a non-inout HW value type, or an aggregate containing HW value types
and `!seq.clock` leaves. This permits probes of clocks and aggregates while
excluding writable or bidirectional hardware references.

Probe references may be exposed through HW module output ports. They must not
appear, directly or nested in an aggregate, on input or inout ports. This keeps
the producer of an observation explicit in the module hierarchy.

## Example

The producer creates a probe handle for `%in` and returns it through an output
port. The consumer receives that handle from an instance and reads its payload.

```mlir
hw.module @Producer(in %in: i8, out p: !probe.ref<i8>) {
  %p = probe.send %in : i8
  hw.output %p : !probe.ref<i8>
}

hw.module @Consumer(in %in: i8, out out: i8) {
  %p = hw.instance "producer" @Producer(in: %in: i8) -> (p: !probe.ref<i8>)
  %value = probe.read %p : <i8>
  hw.output %value : i8
}
```

The Probe dialect intentionally models only read-only observation handles. It
does not prescribe a physical implementation, a hierarchical path encoding, or
writable probe semantics.

## Types

[include "Dialects/ProbeTypes.md"]

## Operations

[include "Dialects/ProbeOps.md"]
