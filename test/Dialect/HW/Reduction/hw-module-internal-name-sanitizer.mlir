// UNSUPPORTED: system-windows
// RUN: circt-reduce %s --include=hw-module-internal-name-sanitizer --test /usr/bin/env --test-arg true --keep-best=0 | FileCheck %s

// CHECK-LABEL: hw.module @TestNamehint
// CHECK-NEXT: %[[C:.+]] = hw.constant
// CHECK-NEXT: %wire = hw.wire
// CHECK-NEXT: %[[XOR:.+]] = comb.xor
// CHECK-NEXT: hw.output
// CHECK-LABEL: hw.module @TestInternal
// CHECK-NEXT: %my_instance.result = hw.instance "my_instance"

hw.module @TestNamehint(in %input1: i8, in %input2: i8, out result: i8) {
  %c42_i8 = hw.constant 42 : i8
  %my_wire = hw.wire %c42_i8 {sv.namehint = "my_descriptive_wire_name"} : i8
  %xor_result = comb.xor %input1, %input2 {sv.namehint = "xor_of_inputs"} : i8
  hw.output %xor_result : i8
}

hw.module @TestInternal(in %in1: i8, in %in2: i8, out out: i8) {
  %result = hw.instance "my_instance" @TestNamehint(input1: %in1: i8, input2: %in2: i8) -> (result: i8)
  hw.output %result : i8
}

