// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects)))' -verify-diagnostics -split-input-file

firrtl.circuit "Missing" {
  firrtl.class @Class(in %input: !firrtl.string) {}
  firrtl.module @Missing() {
    // expected-error @below {{uninitialized input port "input"}}
    %obj = firrtl.object @Class(in input: !firrtl.string)
  }
}

// -----

// Reading an input through an expression is diagnosed before considering
// whether the object can be lowered.
firrtl.circuit "Misuse" {
  firrtl.class @Class(in %input: !firrtl.integer) {}
  firrtl.module @Misuse() {
    %obj = firrtl.object @Class(in input: !firrtl.integer)
    // expected-error @below {{input port may only be used as an assignment destination}}
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Class(in input: !firrtl.integer)>
    %sum = firrtl.integer.add %in, %in : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    firrtl.propassign %in, %sum : !firrtl.integer
  }
}

// -----

// Missing inputs are still diagnosed on an object with a cyclic driver.
firrtl.circuit "CyclicMissing" {
  firrtl.class @Class(in %input: !firrtl.integer, in %missing: !firrtl.integer, out %output: !firrtl.integer) {
    firrtl.propassign %output, %input : !firrtl.integer
  }
  firrtl.module @CyclicMissing() {
    // expected-error @below {{uninitialized input port "missing"}}
    %obj = firrtl.object @Class(in input: !firrtl.integer, in missing: !firrtl.integer, out output: !firrtl.integer)
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Class(in input: !firrtl.integer, in missing: !firrtl.integer, out output: !firrtl.integer)>
    %out = firrtl.object.subfield %obj[output] : !firrtl.class<@Class(in input: !firrtl.integer, in missing: !firrtl.integer, out output: !firrtl.integer)>
    firrtl.propassign %in, %out : !firrtl.integer
  }
}

// -----

// Separate projections of the same input must not hide multiple assignments.
firrtl.circuit "Multiple" {
  firrtl.class @Class(in %input: !firrtl.string) {}
  firrtl.module @Multiple(in %arg: !firrtl.string) {
    %obj = firrtl.object @Class(in input: !firrtl.string)
    %a = firrtl.object.subfield %obj[input] : !firrtl.class<@Class(in input: !firrtl.string)>
    %b = firrtl.object.subfield %obj[input] : !firrtl.class<@Class(in input: !firrtl.string)>
    // expected-error @below {{input port "input" cannot be driven by multiple operations}}
    firrtl.propassign %a, %arg : !firrtl.string
    // expected-note @below {{other driver is here}}
    firrtl.propassign %b, %arg : !firrtl.string
  }
}
