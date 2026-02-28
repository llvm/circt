// RUN: circt-opt %s --arc-lower-clocks-to-funcs --split-input-file --verify-diagnostics

arc.model @NonConstExternalValue io !hw.modty<> {
^bb0(%arg0: !arc.storage<42>):
  %c0_i9001 = hw.constant 0 : i9001
  // expected-note @+1 {{external value defined here:}}
  %0 = comb.add %c0_i9001, %c0_i9001 : i9001
  // expected-note @+1 {{clock tree:}}
  arc.initial {
    // expected-error @+2 {{operation in clock tree uses external value}}
    // expected-note @+1 {{clock trees can only use external constant values}}
    %1 = comb.sub %0, %0 : i9001
  }
}

// -----

func.func private @Victim(!arc.storage<42>)

// expected-warning @below {{Existing model initializer 'Victim' will be overridden.}}
// expected-warning @below {{Existing model finalizer 'Victim' will be overridden.}}
arc.model @ExistingInitializerAndFinalizer io !hw.modty<> initializer @Victim finalizer @Victim {
^bb0(%arg0: !arc.storage<42>):
  arc.initial {}
  arc.final {}
}

// -----

// expected-error @below {{op containing multiple InitialOps is currently unsupported.}}
// expected-error @below {{op containing multiple FinalOps is currently unsupported.}}
arc.model @MultiInitAndPassThrough io !hw.modty<> {
^bb0(%arg0: !arc.storage<1>):
  // expected-note @below {{Conflicting InitialOp:}}
  arc.initial {}
  // expected-note @below {{Conflicting FinalOp:}}
  arc.final {}
  // expected-note @below {{Conflicting InitialOp:}}
  arc.initial {}
  // expected-note @below {{Conflicting FinalOp:}}
  arc.final {}
}
