// RUN: spt-opt %s -split-input-file -verify-diagnostics

"firrtl.module"() ( {
  // expected-error @+1 {{unknown firrtl type}}
  %0 = "firrtl.input"() {name = "in"} : () -> !firrtl.unknowntype
  "firrtl.done"() : () -> ()
}) {name = "MyModule"} : () -> ()

// -----

"firrtl.module"() ( {
  %0 = "firrtl.output"() {name = "out"} : () -> !firrtl.uint
  %1 = "firrtl.input"() {name = "b"} : () -> ui32
  %2 = "firrtl.input"() {name = "d"} : () -> ui16
  // expected-error @+1 {{'firrtl.add' op expected 2 operands, but found 3}}
  %3 = "firrtl.add"(%1, %2, %3) : (ui32, ui16, ui32) -> ui32
  "firrtl.done"() : () -> ()
}) {name = "Top"} : () -> ()

// -----

// expected-error @+2 {{'firrtl.module' op expects regions to end with 'firrtl.done'}}
// expected-note @+1 {{implies 'firrtl.done'}}
"firrtl.module"() ( {
  %0 = "firrtl.output"() {name = "out"} : () -> !firrtl.uint

}) {name = "MyModule"} : () -> ()

// -----


// expected-error @+1 {{'firrtl.module' op requires attribute 'name'}}
"firrtl.module"() ( {
  "firrtl.done"() : () -> ()
}) {no_name = "MyModule"} : () -> ()
