// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @FormatStringSignature
// CHECK-SAME: (%arg0: !sim.fstring) -> !sim.fstring
func.func @FormatStringSignature(%fmt: !moore.format_string) -> !moore.format_string {
  // CHECK: return %arg0 : !sim.fstring
  return %fmt : !moore.format_string
}

// CHECK-LABEL: func.func @FormatStringLiteralReturn
// CHECK-SAME: () -> !sim.fstring
func.func @FormatStringLiteralReturn() -> !moore.format_string {
  // CHECK: %[[FMT:.*]] = sim.fmt.literal "agent11"
  %fmt = moore.fmt.literal "agent11"
  // CHECK: return %[[FMT]] : !sim.fstring
  return %fmt : !moore.format_string
}

// CHECK-LABEL: func.func @FormatStringToString
// CHECK-SAME: (%arg0: !sim.dstring) -> !sim.dstring
func.func @FormatStringToString(%arg0: !moore.string) -> !moore.string {
  // CHECK: %[[FMT:.*]] = sim.fmt.string %arg0
  %fmt = moore.fmt.string %arg0
  // CHECK: %[[STR:.*]] = builtin.unrealized_conversion_cast %[[FMT]] : !sim.fstring to !sim.dstring
  %str = moore.fstring_to_string %fmt
  // CHECK: return %[[STR]] : !sim.dstring
  return %str : !moore.string
}

// CHECK-LABEL: func.func @FormatLiteralToString
// CHECK-SAME: () -> !sim.dstring
func.func @FormatLiteralToString() -> !moore.string {
  // CHECK: %[[FMT:.*]] = sim.fmt.literal "agent11"
  %fmt = moore.fmt.literal "agent11"
  // CHECK: %[[STR:.*]] = builtin.unrealized_conversion_cast %[[FMT]] : !sim.fstring to !sim.dstring
  %str = moore.fstring_to_string %fmt
  // CHECK: return %[[STR]] : !sim.dstring
  return %str : !moore.string
}

// CHECK: llhd.global_signal @GlobalStringInit : !sim.dstring init {
// CHECK: %[[GLOBAL_FMT:.*]] = sim.fmt.literal "global"
// CHECK: %[[GLOBAL_STR:.*]] = builtin.unrealized_conversion_cast %[[GLOBAL_FMT]] : !sim.fstring to !sim.dstring
// CHECK: llhd.yield %[[GLOBAL_STR]] : !sim.dstring
moore.global_variable @GlobalStringInit : !moore.string init {
  %fmt = moore.fmt.literal "global"
  %str = moore.fstring_to_string %fmt
  moore.yield %str : !moore.string
}

// CHECK-LABEL: func.func @StringToFormatStringConversion
// CHECK-SAME: (%arg0: !sim.dstring) -> !sim.dstring
func.func @StringToFormatStringConversion(%input: !moore.string) -> !moore.string {
  // CHECK: %[[FMT:.*]] = sim.fmt.string %arg0 : !sim.dstring
  %fmt = moore.conversion %input : !moore.string -> !moore.format_string
  // CHECK: %[[STR:.*]] = builtin.unrealized_conversion_cast %[[FMT]] : !sim.fstring to !sim.dstring
  %str = moore.fstring_to_string %fmt
  // CHECK: return %[[STR]] : !sim.dstring
  return %str : !moore.string
}

// CHECK-LABEL: func.func @FormatStringToStringConversion
// CHECK-SAME: (%arg0: !sim.dstring) -> !sim.dstring
func.func @FormatStringToStringConversion(%input: !moore.string) -> !moore.string {
  // CHECK: %[[FMT:.*]] = sim.fmt.string %arg0
  %fmt = moore.fmt.string %input
  // CHECK: %[[STR:.*]] = builtin.unrealized_conversion_cast %[[FMT]] : !sim.fstring to !sim.dstring
  %str = moore.conversion %fmt : !moore.format_string -> !moore.string
  // CHECK: return %[[STR]] : !sim.dstring
  return %str : !moore.string
}

// CHECK-LABEL: hw.module @FormatStringModulePorts
// CHECK-SAME: in %fmt_in : !sim.fstring
// CHECK-SAME: out fmt_out : !sim.fstring
moore.module @FormatStringModulePorts(in %fmt_in : !moore.format_string, out fmt_out : !moore.format_string) {
  // CHECK: hw.output %fmt_in : !sim.fstring
  moore.output %fmt_in : !moore.format_string
}

// CHECK-LABEL: hw.module @FormatStringInstTop
// CHECK-SAME: in %fmt_in : !sim.fstring
// CHECK-SAME: out fmt_out : !sim.fstring
moore.module @FormatStringInstTop(in %fmt_in : !moore.format_string, out fmt_out : !moore.format_string) {
  // CHECK: hw.instance "child" @FormatStringInstChild(fmt_in: %fmt_in: !sim.fstring) -> (fmt_out: !sim.fstring)
  %child.fmt_out = moore.instance "child" @FormatStringInstChild(fmt_in: %fmt_in : !moore.format_string) -> (fmt_out: !moore.format_string)
  // CHECK: hw.output %child.fmt_out : !sim.fstring
  moore.output %child.fmt_out : !moore.format_string
}

// CHECK-LABEL: hw.module private @FormatStringInstChild
moore.module private @FormatStringInstChild(in %fmt_in : !moore.format_string, out fmt_out : !moore.format_string) {
  // CHECK: hw.output %fmt_in : !sim.fstring
  moore.output %fmt_in : !moore.format_string
}
