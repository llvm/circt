
// RUN: circt-synth %s --allow-unregistered-dialect
// RUN: not circt-synth %s

hw.module @and(in %a: i2, in %b: i2, in %c: i2) {
  "foo"(%a) : (i2) -> ()
}
