// RUN: circt-opt %s --arc-lower-processes --split-input-file --verify-diagnostics

// Test 1: Process with block operands on wait (should be rejected)
arc.model @BlockOperandsOnWait io !hw.modty<output x : i42> {
^bb0(%arg0: !arc.storage):
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c0_i42 = hw.constant 0 : i42
  %c42_i42 = hw.constant 42 : i42
  
  // expected-error @+1 {{processes with block operands on llhd.wait are not supported}}
  %1 = llhd.process -> i42 {
    llhd.wait yield (%c0_i42 : i42), delay %time, ^bb1(%c42_i42 : i42)
  ^bb1(%arg1: i42):
    llhd.halt %arg1 : i42
  }
}

// -----

// Test 2: Process with unsupported terminator
arc.model @UnsupportedTerminator io !hw.modty<output x : i42> {
^bb0(%arg0: !arc.storage):
  %c0_i42 = hw.constant 0 : i42
  
  // expected-error @+1 {{unsupported terminator in process}}
  %1 = llhd.process -> i42 {
    cf.br ^bb1
  ^bb1:
    llhd.halt %c0_i42 : i42
  }
}

// -----

// Test 3: Process with no results (should be rejected)
arc.model @NoResults io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  %time = llhd.constant_time <1ns, 0d, 0e>
  
  // expected-error @+1 {{process must have exactly one result}}
  llhd.process {
    llhd.wait delay %time, ^bb1
  ^bb1:
    llhd.halt
  }
}

// -----

// Test 4: Process with multiple results (should be rejected)
arc.model @MultipleResults io !hw.modty<output x : i42, output y : i8> {
^bb0(%arg0: !arc.storage):
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c0_i42 = hw.constant 0 : i42
  %c0_i8 = hw.constant 0 : i8

  // expected-error @+1 {{processes with multiple results are not yet supported}}
  %1:2 = llhd.process -> i42, i8 {
    llhd.wait yield (%c0_i42, %c0_i8 : i42, i8), delay %time, ^bb1
  ^bb1:
    llhd.halt %c0_i42, %c0_i8 : i42, i8
  }
}

