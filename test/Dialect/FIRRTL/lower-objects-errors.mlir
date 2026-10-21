// RUN: circt-opt %s -firrtl-lower-objects -verify-diagnostics -split-input-file

firrtl.circuit "Missing" {
  firrtl.class @Class(in %input: !firrtl.string) {}
  firrtl.module @Missing() {
    // expected-error @below {{uninitialized input port "input"}}
    %obj = firrtl.object @Class(in input: !firrtl.string)
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

// -----

firrtl.circuit "Cycle" {
  firrtl.class @Class(in %input: !firrtl.string, out %output: !firrtl.string) {
    firrtl.propassign %output, %input : !firrtl.string
  }
  // expected-error @below {{cyclic object input dependencies}}
  firrtl.module @Cycle() {
    %obj = firrtl.object @Class(in input: !firrtl.string, out output: !firrtl.string)
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Class(in input: !firrtl.string, out output: !firrtl.string)>
    %out = firrtl.object.subfield %obj[output] : !firrtl.class<@Class(in input: !firrtl.string, out output: !firrtl.string)>
    firrtl.propassign %in, %out : !firrtl.string
  }
}
