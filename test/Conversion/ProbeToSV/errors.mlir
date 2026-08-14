// RUN: circt-opt %s --lower-probe-to-sv --verify-diagnostics --split-input-file

// expected-error @below {{public Probe output ports require an external Probe ABI and cannot be lowered to an SV XMR}}
hw.module @PublicOutput(in %in: i8, out p: !probe.ref<i8>) {
  %p = probe.send %in : i8
  hw.output %p : !probe.ref<i8>
}

// -----

// expected-error @below {{Probe output ports on external or generated modules are not supported by Probe-to-SV lowering}}
hw.module.extern private @ExternalProbe(out p: !probe.ref<i8>)

// -----

hw.generator.schema @Schema, "Schema", []
// expected-error @below {{Probe output ports on external or generated modules are not supported by Probe-to-SV lowering}}
hw.module.generated private @GeneratedProbe, @Schema(out p: !probe.ref<i8>)

// -----

hw.module @ClockPayload(in %clock: !seq.clock, out out: !seq.clock) {
  // expected-error @+1 {{Probe-to-SV lowering does not support payload type '!seq.clock'; lower it to an HW value type before this pass}}
  %p = probe.send %clock : !seq.clock
  %read = probe.read %p : <!seq.clock>
  hw.output %read : !seq.clock
}

// -----

hw.module @AggregateClockPayload(
    in %input: !hw.struct<clock: !seq.clock, data: i8>,
    out output: !hw.struct<clock: !seq.clock, data: i8>) {
  // expected-error @+1 {{Probe-to-SV lowering does not support payload type '!hw.struct<clock: !seq.clock, data: i8>'; lower it to an HW value type before this pass}}
  %p = probe.send %input : !hw.struct<clock: !seq.clock, data: i8>
  %read = probe.read %p : <!hw.struct<clock: !seq.clock, data: i8>>
  hw.output %read : !hw.struct<clock: !seq.clock, data: i8>
}

// -----

hw.module private @Leaf(in %in: i8, out p: !probe.ref<i8>) {
  %p = probe.send %in : i8
  hw.output %p : !probe.ref<i8>
}

hw.module private @Middle(in %in: i8, out p: !probe.ref<i8>) {
  %p = hw.instance "leaf" @Leaf(in: %in: i8) -> (p: !probe.ref<i8>)
  // expected-error @below {{Probe output must be driven directly by probe.send; forwarding Probe refs across multiple module levels is not supported}}
  hw.output %p : !probe.ref<i8>
}

// -----

// expected-error @below {{nested Probe refs in module ports are not supported by Probe-to-SV lowering}}
hw.module private @NestedOutput(out p: !hw.struct<p: !probe.ref<i8>>) {
  %p = builtin.unrealized_conversion_cast to !hw.struct<p: !probe.ref<i8>>
  hw.output %p : !hw.struct<p: !probe.ref<i8>>
}

// -----

hw.module @GenericPropagation(in %in: i8, out out: i8) {
  %p = probe.send %in : i8
  // expected-error @below {{Probe refs may only flow through probe.send, probe.read, hw.output, and direct hw.instance results during Probe-to-SV lowering}}
  %forwarded = builtin.unrealized_conversion_cast %p : !probe.ref<i8> to !probe.ref<i8>
  %read = probe.read %forwarded : <i8>
  hw.output %read : i8
}

// -----

hw.module private @DoNotPrintChild(in %in: i8,
                                   out p: !probe.ref<i8>) {
  %p = probe.send %in : i8
  hw.output %p : !probe.ref<i8>
}

hw.module @DoNotPrintInstance(in %in: i8, out out: i8) {
  %p = hw.instance "child" @DoNotPrintChild(in: %in: i8) ->
      (p: !probe.ref<i8>) {doNotPrint}
  // expected-error @+1 {{Probe input cannot originate from an hw.instance marked doNotPrint}}
  %read = probe.read %p : <i8>
  hw.output %read : i8
}
