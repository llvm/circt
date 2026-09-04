// RUN: circt-opt %s --lower-probe-to-sv --verify-diagnostics --split-input-file

// expected-error @below {{Probe-to-SV lowering cannot remove a Probe output from a public module without a defined external ABI}}
hw.module @PublicOutput(in %in: i8, out p: !probe.ref<i8>) {
  %p = probe.send %in : i8
  hw.output %p : !probe.ref<i8>
}

// -----

// expected-error @below {{Probe refs on external or generated module ports require a defined Probe ABI}}
hw.module.extern private @ExternalProbe(out p: !probe.ref<i8>)

// -----

hw.generator.schema @Schema, "Schema", []
// expected-error @below {{Probe refs on external or generated module ports require a defined Probe ABI}}
hw.module.generated private @GeneratedProbe, @Schema(out p: !probe.ref<i8>)

// -----

hw.module @ClockPayload(in %clock: !seq.clock, out out: !seq.clock) {
  // expected-error @+1 {{Probe-to-SV lowering requires an HW value payload, but got '!seq.clock'}}
  %p = probe.send %clock : !seq.clock
  %read = probe.read %p : <!seq.clock>
  hw.output %read : !seq.clock
}

// -----

hw.module @AggregateClockPayload(
    in %input: !hw.struct<clock: !seq.clock, data: i8>,
    out output: !hw.struct<clock: !seq.clock, data: i8>) {
  // expected-error @+1 {{Probe-to-SV lowering requires an HW value payload, but got '!hw.struct<clock: !seq.clock, data: i8>'}}
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
  // expected-error @below {{the Probe dialect requires a Probe output to be driven directly by probe.send; multi-level forwarding is not supported}}
  hw.output %p : !probe.ref<i8>
}

// -----

// expected-error @below {{the Probe dialect does not support nested Probe refs in module ports}}
hw.module private @NestedOutput(out p: !hw.struct<p: !probe.ref<i8>>) {
  %p = builtin.unrealized_conversion_cast to !hw.struct<p: !probe.ref<i8>>
  hw.output %p : !hw.struct<p: !probe.ref<i8>>
}

// -----

hw.module @GenericPropagation(in %in: i8, out out: i8) {
  %p = probe.send %in : i8
  // expected-error @below {{the Probe dialect only permits Probe refs to flow through probe.send, probe.read, hw.output, and direct hw.instance results}}
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
  // expected-error @+1 {{Probe-to-SV lowering cannot create an XMR through an hw.instance marked doNotPrint}}
  %read = probe.read %p : <i8>
  hw.output %read : i8
}
