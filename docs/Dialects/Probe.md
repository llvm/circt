# Probe Dialect

The Probe dialect provides SSA handles for observation of hardware values. A
probe handle can be passed through module outputs and used where access to the
observed value is needed, without representing that relationship as ordinary
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

A probe reference provides read-only access to an observed hardware value.
Its payload type must be accepted by the HW probe type classification. The
current implementation supports integer and non-inout HW value types,
`!seq.clock`, and HW aggregates recursively containing supported types.
Dialects may extend the set of supported payload types through
`hw::ProbeTypeDialectInterface`. Writable or bidirectional hardware references
are not supported.

Probe references may be exposed through HW module output ports. They must not
appear, directly or nested in an aggregate, on input or inout ports. Frontends
must legalize such cases to ordinary ports, XMRs, or another suitable
representation before creating Probe dialect IR. This keeps the producer of an
observation explicit in the module hierarchy. This restriction simplifies
compilation for a wide range of backends.

## Supported Placement and Propagation

The initial Probe dialect supports the following narrow path: `probe.send` and
`probe.read` appear directly in an `hw.module` body, a probe reference is
returned through an output port, and the corresponding `hw.instance` result is
consumed by `probe.read` in the enclosing module body. This is the only
cross-module propagation path for which the dialect currently defines
semantics.

The Probe operations themselves cannot be placed in procedural regions such as
`hw.triggered`, or in other nested regions. Probe handles must not be created or
read inside those regions. Procedural logic should consume an ordinary hardware
value obtained by reading a probe in the enclosing module body. The dialect
does not define semantics for passing a probe reference across a region
boundary; such uses should be legalized before creating Probe dialect IR.

Probe references propagated through instance-choice operations are currently
unsupported. The dialect does not define how a probe reference is associated
with the selected module, so FIRRTL lowering must diagnose or reject such uses
until the required semantics are specified.

Probe references are not supported as HW inner-symbol targets. This applies
both to a `!probe.ref<T>` value and to any field containing one. Other
propagation paths through generic operations are similarly outside the current
Probe dialect contract, even if those operations accept the type through a
generic type constraint such as `AnyType`.

## Example

The producer creates a probe handle for `%in` and returns it through an output
port. The consumer receives the handle from an instance and reads the observed
value through it.

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

`probe.send` accepts any SSA value whose type is valid as the payload of a probe
reference, including the result of an expression. It produces only the probe
reference; ordinary dataflow consumers continue to use the original SSA value:

```mlir
%value = comb.xor %a, %b : i8
%ref = probe.send %value : i8
%next = comb.xor %value, %c : i8
```

A probe observes the value passed to `probe.send`, not a particular SSA
definition or expression representation. Optimizations may rewrite the
producer, retarget the probe, or remove an unused probe handle as long as
`probe.read` observes the same value. The observed value does not need to have
an explicit name in the IR. A backend may generate a name when required, but
automatically generated names are implementation details and are not
guaranteed to remain stable across compiler runs or IR transformations.

External-module probe ABIs are presently outside the scope of this dialect
definition.

## Types

[include "Dialects/ProbeTypes.md"]

## Operations

[include "Dialects/ProbeOps.md"]
