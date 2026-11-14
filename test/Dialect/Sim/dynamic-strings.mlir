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
// CHECK: hw.constant 13 : i32
hw.module @string_length_constant_fold(out res: i32) {
  %str = sim.string.literal "Hello, World!"
  %len = sim.string.length %str
  hw.output %len : i32
}

// CHECK-LABEL: hw.module @string_length_empty_fold
// CHECK: hw.constant 0 : i32
hw.module @string_length_empty_fold(out res: i32) {
  %str = sim.string.literal ""
  %len = sim.string.length %str
  hw.output %len : i32
}

// CHECK-LABEL: hw.module @string_concat_empty_operands
// CHECK: sim.string.literal ""
hw.module @string_concat_empty_operands(out res: !sim.dstring) {
  %concat = sim.string.concat ()
  hw.output %concat : !sim.dstring
}

// CHECK-LABEL: hw.module @string_concat_single_operand
// CHECK: sim.string.literal "Single"
hw.module @string_concat_single_operand(out res: !sim.dstring) {
  %str = sim.string.literal "Single"
  %concat = sim.string.concat (%str)
  hw.output %concat : !sim.dstring
}

// CHECK-LABEL: hw.module @string_concat_multiple_literals
// CHECK: sim.string.literal "OneTwoThree"
hw.module @string_concat_multiple_literals(out res: !sim.dstring) {
  %str1 = sim.string.literal "One"
  %str2 = sim.string.literal "Two"
  %str3 = sim.string.literal "Three"
  %concat = sim.string.concat (%str1, %str2, %str3)
  hw.output %concat : !sim.dstring
}

// Concat with a non-constant operand should not crash during folding.
// CHECK-LABEL: hw.module @string_concat_non_constant
// CHECK: sim.string.concat
hw.module @string_concat_non_constant(in %s: !sim.dstring, out res: !sim.dstring) {
  %str = sim.string.literal "Hello"
  %concat = sim.string.concat (%str, %s)
  hw.output %concat : !sim.dstring
}

// CHECK-LABEL: hw.module @convert_int_to_string_0
// CHECK: sim.string.literal "Test"
hw.module @convert_int_to_string_0(out res: !sim.dstring) {
  %a = hw.constant 1415934836 : i32
  %str = sim.string.int_to_string %a : i32
  hw.output %str : !sim.dstring
}

// CHECK-LABEL: hw.module @convert_int_to_string_1
// CHECK: sim.string.literal "Hlo"
hw.module @convert_int_to_string_1(out res: !sim.dstring) {
  %a0 = hw.constant 310939249775 : i40
  %a1 = hw.constant 1095233437695 : i40
  %a2 = hw.constant 1099494916095 : i40
  %a3 = comb.and %a0, %a1 : i40
  %a4 = comb.and %a3, %a2 : i40
  %str = sim.string.int_to_string %a4 : i40
  hw.output %str : !sim.dstring
}

// CHECK-LABEL: hw.module @convert_int_to_string_3
// CHECK: hw.constant 3 : i32
hw.module @convert_int_to_string_3(out res: i32) {
  %a0 = hw.constant 310939249775 : i40
  %a1 = hw.constant 1095233437695 : i40
  %a2 = hw.constant 1099494916095 : i40
  %a3 = comb.and %a0, %a1 : i40
  %a4 = comb.and %a3, %a2 : i40
  %str = sim.string.int_to_string %a4 : i40
  %len = sim.string.length %str
  hw.output %len : i32
}

// CHECK-LABEL: hw.module @convert_int_to_string_4
// CHECK: sim.string.literal ""
hw.module @convert_int_to_string_4(out res: !sim.dstring) {
  %a0 = hw.constant 0 : i40
  %str = sim.string.int_to_string %a0 : i40
  hw.output %str : !sim.dstring
}

// CHECK-LABEL: hw.module @string_substr_1
// CHECK: sim.string.literal "Hel"
hw.module @string_substr_1(out res: !sim.dstring) {
  %str = sim.string.literal "Hello World"
  %i = hw.constant -1 : i32
  %j = hw.constant 3 : i32
  %sub = sim.string.substr %str[%i : %j]
  hw.output %sub : !sim.dstring
}

// CHECK-LABEL: hw.module @string_substr_2
// CHECK: sim.string.literal ""
hw.module @string_substr_2(out res: !sim.dstring) {
  %str = sim.string.literal "Hello World"
  %i = hw.constant 2 : i32
  %j = hw.constant 1 : i32
  %sub = sim.string.substr %str[%i : %j]
  hw.output %sub : !sim.dstring
}

// CHECK-LABEL: hw.module @string_substr_3
// CHECK: sim.string.literal "llo World"
hw.module @string_substr_3(out res: !sim.dstring) {
  %str = sim.string.literal "Hello World"
  %i = hw.constant 2 : i32
  %j = hw.constant 20 : i32
  %sub = sim.string.substr %str[%i : %j]
  hw.output %sub : !sim.dstring
}

// CHECK-LABEL: hw.module @string_substr_4
// CHECK: sim.string.literal "ll"
hw.module @string_substr_4(out res: !sim.dstring) {
  %str = sim.string.literal "Hello World"
  %i = hw.constant 2 : i32
  %j = hw.constant 4 : i32
  %sub = sim.string.substr %str[%i : %j]
  hw.output %sub : !sim.dstring
}

// CHECK-LABEL: hw.module @string_substr_5
// CHECK: sim.string.literal "lo World"
hw.module @string_substr_5(out res: !sim.dstring) {
  %str = sim.string.literal "Hello World"
  %i = hw.constant 3 : i32
  %sub = sim.string.substr %str[%i]
  hw.output %sub : !sim.dstring
}

// CHECK-LABEL: hw.module @string_substr_6
hw.module @string_substr_6(in %end: i32, out res: !sim.dstring) {
  // CHECK: [[STR:%.+]] = sim.string.literal "Hello World"
  // CHECK: [[START:%.+]] = hw.constant 3 : i32
  // CHECK: [[SUBSTR:%.+]] = sim.string.substr [[STR]][[[START]] : %end]
  %str = sim.string.literal "Hello World"
  %start = hw.constant 3 : i32
  %sub = sim.string.substr %str[%start : %end]
  hw.output %sub : !sim.dstring
}
