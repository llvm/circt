// RUN: circt-opt %s --arc-lower-clocks-to-funcs --split-input-file --verify-diagnostics

arc.model @NonConstExternalValue io !hw.modty<> {
^bb0(%arg0: !arc.storage<42>):
  %c0_i9001 = hw.constant 0 : i9001
  // expected-note @+1 {{external value defined here:}}
  %0 = comb.add %c0_i9001, %c0_i9001 : i9001
  // expected-note @+1 {{clock tree:}}
  arc.passthrough {
    // expected-error @+2 {{operation in clock tree uses external value}}
    // expected-note @+1 {{clock trees can only use external constant values}}
    %1 = comb.sub %0, %0 : i9001
  }
}

// -----

func.func @VictimInit(%arg0: !arc.storage<42>) {
  return
}

// expected-warning @below {{Existing model initializer 'VictimInit' will be overriden.}}
arc.model @ExistingInit io !hw.modty<> initializer @VictimInit {
^bb0(%arg0: !arc.storage<42>):
  arc.initial {
    %c0_i9001 = hw.constant 0 : i9001
    %1 = comb.sub %c0_i9001, %c0_i9001 : i9001
  }
}