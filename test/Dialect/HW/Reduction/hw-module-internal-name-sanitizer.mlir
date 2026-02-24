// UNSUPPORTED: system-windows
// RUN: circt-reduce %s --include=hw-module-internal-name-sanitizer --test /usr/bin/env --test-arg true --keep-best=0 | FileCheck %s

// CHECK-LABEL: hw.module @Module1
// CHECK-NEXT: %wire = hw.wire
// CHECK-NEXT: hw.output

hw.module @Module1(in %input: i8, out result: i8) {
  %my_descriptive_wire_name = hw.wire %input : i8
  hw.output %my_descriptive_wire_name : i8
}
