// RUN: circt-opt %s --canonicalize | FileCheck --strict-whitespace %s

// CHECK-LABEL: hw.module @string_literal_basic
// CHECK: sim.string.literal "Hello, World!"
hw.module @string_literal_basic(out res: !sim.dstring) {
  %str = sim.string.literal "Hello, World!"
  hw.output %str : !sim.dstring
}

// CHECK-LABEL: hw.module @string_literal_empty
// CHECK: sim.string.literal ""
hw.module @string_literal_empty(out res: !sim.dstring) {
  %str = sim.string.literal ""
  hw.output %str : !sim.dstring
}

// CHECK-LABEL: hw.module @string_length_constant_fold
// CHECK: hw.constant 13 : i64
hw.module @string_length_constant_fold(out res: i64) {
  %str = sim.string.literal "Hello, World!"
  %len = sim.string.length %str
  hw.output %len : i64
}

// CHECK-LABEL: hw.module @string_length_empty_fold
// CHECK: hw.constant 0 : i64
hw.module @string_length_empty_fold(out res: i64) {
  %str = sim.string.literal ""
  %len = sim.string.length %str
  hw.output %len : i64
}

// CHECK-LABEL: hw.module @string_concat_empty_operands
// CHECK: sim.string.literal ""
hw.module @string_concat_empty_operands(out res: !sim.dstring) {
  %concat = sim.string.concat()
  hw.output %concat : !sim.dstring
}

// CHECK-LABEL: hw.module @string_concat_single_operand
// CHECK: sim.string.literal "Single"
hw.module @string_concat_single_operand(out res: !sim.dstring) {
  %str = sim.string.literal "Single"
  %concat = sim.string.concat(%str)
  hw.output %concat : !sim.dstring
}

// CHECK-LABEL: hw.module @string_concat_multiple_literals
// CHECK: sim.string.literal "OneTwoThree"
hw.module @string_concat_multiple_literals(out res: !sim.dstring) {
  %str1 = sim.string.literal "One"
  %str2 = sim.string.literal "Two"
  %str3 = sim.string.literal "Three"
  %concat = sim.string.concat(%str1, %str2, %str3)
  hw.output %concat : !sim.dstring
}
