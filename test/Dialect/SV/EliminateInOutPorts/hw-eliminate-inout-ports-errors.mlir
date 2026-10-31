// RUN: circt-opt --split-input-file -allow-unregistered-dialect --hw-eliminate-inout-ports -verify-diagnostics %s

hw.module @unsupported(in %a: !sv.net<i42>) {
  // expected-error @+1 {{uses sv.net port "a" but the operation itself is unsupported.}}
  "foo.bar" (%a) : (!sv.net<i42>) -> ()
}

// -----

// expected-error @+1 {{multiple writers of inout port "a" is unsupported.}}
hw.module @multipleWriters(in %a: !sv.net<i42>) {
  %0 = hw.constant 0 : i42
  sv.assign %a, %0 : !sv.net<i42>, i42
  sv.assign %a, %0 : !sv.net<i42>, i42
}
