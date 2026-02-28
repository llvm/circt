// UNSUPPORTED: system-windows
// RUN: circt-reduce %s --include=sv-namehint-remover --test /usr/bin/env --test-arg true --keep-best=0 | FileCheck %s --implicit-check-not=sv.namehint

// CHECK-LABEL: hw.module @TestNamehint

hw.module @TestNamehint(in %input1: i8, in %input2: i8, out result: i8) {
  %c42_i8 = hw.constant 42 : i8
  %my_wire = hw.wire %c42_i8 {sv.namehint = "my_descriptive_wire_name"} : i8
  %xor_result = comb.xor %input1, %input2 {sv.namehint = "xor_of_inputs"} : i8
  hw.output %xor_result : i8
}

