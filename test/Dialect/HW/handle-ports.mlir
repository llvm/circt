// RUN: circt-opt %s --verify-diagnostics --split-input-file

// expected-error @below {{sv.net handles are not supported on output ports}}
hw.module @NetOutput(out p: !sv.net<i4>) {
}

// -----

// expected-error @below {{sv.net handles are not supported on output ports}}
hw.module @AggregateNetOutput(out p: !hw.struct<n: !sv.net<i4>>) {
}

// -----

hw.module @NetInput(in %p: !sv.net<i4>) {
}
