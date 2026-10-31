// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-promote-objects)))' -verify-diagnostics

firrtl.circuit "InputRead" {
  firrtl.extclass @Input(in value: !firrtl.string)
  firrtl.module @InputRead(in %arg: !firrtl.string) {
    %obj = firrtl.object @Input(in value: !firrtl.string)
    // expected-error @below {{input port may only be used as an assignment destination}}
    %field = firrtl.object.subfield %obj[value] : !firrtl.class<@Input(in value: !firrtl.string)>
    %read = firrtl.string.concat %field, %arg : !firrtl.string
    firrtl.propassign %field, %arg : !firrtl.string
  }
}

firrtl.circuit "DuplicateDrivers" {
  firrtl.extclass @Input(in value: !firrtl.string)
  firrtl.module @DuplicateDrivers(in %arg: !firrtl.string) {
    %obj = firrtl.object @Input(in value: !firrtl.string)
    %first = firrtl.object.subfield %obj[value] : !firrtl.class<@Input(in value: !firrtl.string)>
    %second = firrtl.object.subfield %obj[value] : !firrtl.class<@Input(in value: !firrtl.string)>
    // expected-error @below {{input port "value" cannot be driven by multiple operations}}
    firrtl.propassign %first, %arg : !firrtl.string
    // expected-note @below {{other driver is here}}
    firrtl.propassign %second, %arg : !firrtl.string
  }
}

firrtl.circuit "MissingDriver" {
  firrtl.extclass @Input(in value: !firrtl.string)
  firrtl.module @MissingDriver() {
    // expected-error @below {{uninitialized input port "value"}}
    %obj = firrtl.object @Input(in value: !firrtl.string)
  }
}
