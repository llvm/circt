// RUN: circt-opt %s --arc-lower-clocks-to-funcs --split-input-file --verify-diagnostics

arc.model @NonPureExternalValue io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  %0 = arc.as_context %arg0 : !arc.storage
  // expected-note @below {{external value defined here:}}
  %1 = arc.current_time %0
  // expected-error @below {{clock tree uses external value}}
  // expected-note @below {{clock trees can only use pure external values}}
  arc.initial {
    %2 = comb.add %1, %1 : i64
  }
}

// -----

func.func private @Victim(!arc.storage)

// expected-warning @below {{Existing model initializer 'Victim' will be overridden.}}
// expected-warning @below {{Existing model finalizer 'Victim' will be overridden.}}
arc.model @ExistingInitializerAndFinalizer io !hw.modty<> initializer @Victim finalizer @Victim {
^bb0(%arg0: !arc.storage):
  arc.initial {}
  arc.final {}
}

// -----

// expected-error @below {{op containing multiple InitialOps is currently unsupported.}}
// expected-error @below {{op containing multiple FinalOps is currently unsupported.}}
arc.model @MultiInitAndPassThrough io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  // expected-note @below {{Conflicting InitialOp:}}
  arc.initial {}
  // expected-note @below {{Conflicting FinalOp:}}
  arc.final {}
  // expected-note @below {{Conflicting InitialOp:}}
  arc.initial {}
  // expected-note @below {{Conflicting FinalOp:}}
  arc.final {}
}
