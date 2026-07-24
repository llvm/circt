# Probe Dialect

The Probe dialect provides SSA handles for by-name observation of hardware
values. A probe handle can be passed through module outputs and read where the
observed value is needed, without representing that connection as ordinary
hardware dataflow or committing to a particular hierarchical path
representation.

[TOC]

## Rationale

Hardware IR often needs to preserve an observation relationship across module
boundaries. Routing the observed value through ordinary ports changes the
module interface and may introduce unnecessary data dependencies. The Probe
dialect represents this relationship explicitly: `!probe.ref<T>` is a
read-only handle to a value of type `T`, created by `probe.send` and observed
by `probe.read`. A read does not require the compiler to materialize ordinary
hardware dataflow from the probe origin to the read site.

A probe reference provides access to the value of observed hardware, not
writable access to the hardware object itself. It cannot be driven or used for
assignment, force, or release. Its payload must be a non-inout HW value type,
or an aggregate containing HW value types and `!seq.clock` leaves. This permits
probes of clocks and aggregates while excluding writable or bidirectional
hardware references.

Probe references may be exposed through HW module output ports. They must not
appear, directly or nested in an aggregate, on input or inout ports. Frontends
must legalize such cases to ordinary ports, XMRs, or another suitable
representation before creating Probe dialect IR. This keeps the producer of an
observation explicit in the module hierarchy.

## Example

The producer creates a probe handle for `%in` and returns it through an output
port. Since the producer does not use the forwarded result, this is a
probe-only tap. The consumer receives the handle from an instance and reads its
payload by name.

```mlir
hw.module @Producer(in %in: i8, out p: !probe.ref<i8>) {
  %forwarded, %p = probe.send %in : i8
  hw.output %p : !probe.ref<i8>
}

hw.module @Consumer(in %in: i8, out out: i8) {
  %p = hw.instance "producer" @Producer(in: %in: i8) -> (p: !probe.ref<i8>)
  %value = probe.read %p : <i8>
  hw.output %value : i8
}
```

`probe.send` accepts any SSA value with a valid probe element type, including
the result of an expression. Its forwarded result represents an explicit tap
in ordinary dataflow:

```mlir
%value = comb.xor %a, %b : i8
%forwarded, %ref = probe.send %value : i8
%next = comb.xor %forwarded, %c : i8
```

Optimizations must not replace uses of `%forwarded` with `%value` or otherwise
bypass the tap. A backend that requires the observed value to have a name may
materialize an anchor for `probe.send`.

The Probe dialect intentionally models only read-only observation handles. It
does not prescribe a physical implementation, a hierarchical path encoding, or
writable probe semantics. External-module probe ABIs and bind-layer capture
legalization are also outside the scope of this dialect definition.

## Types

[include "Dialects/ProbeTypes.md"]

## Operations

[include "Dialects/ProbeOps.md"]
