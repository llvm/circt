// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

sv.macro.decl @SYNTHESIS
sv.macro.decl @PRINTF_COND_
sv.func private @"__circt_lib_logging::FileDescriptor::get"(in %name : !hw.string, out fd : i32 {sv.func.explicitly_returned}) attributes {verilogName = "__circt_lib_logging::FileDescriptor::get"}

// CHECK-LABEL: hw.module @print_io
hw.module @print_io(in %clk: !seq.clock, in %cond: i1, in %val: i8) {
  %lit = sim.fmt.literal "v="
  %hex = sim.fmt.hex %val, isUpper false : i8
  %nl = sim.fmt.literal "\0A"
  %msg = sim.fmt.concat (%lit, %hex, %nl)
  %signedVal = sim.cast.signed %val : i8
  %dec = sim.fmt.dec %signedVal signed : i8
  %t = sim.time
  %tmsg = sim.fmt.concat (%dec, %nl)
  // CHECK-DAG: sv.system.time : i64

  // CHECK: sv.ifdef @SYNTHESIS {
  // CHECK-NEXT: } else {
  // CHECK-NEXT: [[CLK0:%.+]] = seq.from_clock %clk
  // CHECK-NEXT: sv.always posedge [[CLK0]] {
  // CHECK: sv.if %{{.+}} {
  // CHECK-NEXT:   sv.fwrite %{{.+}}, "v=%x\0A"(%val) : i8
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  sim.print %msg on %clk if %cond {usePrintfCond = true}

  %fd = sim.get_file "out_%0d.log"(%val) : (i8) -> i32
  // CHECK: [[NAME:%.+]] = sv.sformatf "out_%0d.log"(%val) : i8
  // CHECK-NEXT: [[FD:%.+]] = sv.func.call @"__circt_lib_logging::FileDescriptor::get"([[NAME]]) : (!hw.string) -> i32

  // CHECK: sv.ifdef @SYNTHESIS {
  // CHECK-NEXT: } else {
  // CHECK-NEXT: [[CLK1:%.+]] = seq.from_clock %clk
  // CHECK-NEXT: sv.always posedge [[CLK1]] {
  // CHECK-NEXT: sv.if %cond {
  // CHECK-NEXT:   sv.fwrite [[FD]], "v=%x\0A"(%val) : i8
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  sim.print %msg to %fd on %clk if %cond

  // CHECK: sv.ifdef @SYNTHESIS {
  // CHECK-NEXT: } else {
  // CHECK-NEXT: %{{.+}} = seq.from_clock %clk
  // CHECK-NEXT: sv.always posedge %{{.+}} {
  // CHECK-NEXT: sv.if %cond {
  // CHECK-NEXT:   sv.fwrite [[FD]], "%d\0A"(%{{.+}}) : i8
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  sim.print %tmsg to %fd on %clk if %cond

  // CHECK: sv.ifdef @SYNTHESIS {
  // CHECK-NEXT: } else {
  // CHECK-NEXT: [[CLK2:%.+]] = seq.from_clock %clk
  // CHECK-NEXT: sv.always posedge [[CLK2]] {
  // CHECK-NEXT: sv.if %cond {
  // CHECK-NEXT:   sv.fflush fd [[FD]]
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  sim.fflush %fd on %clk if %cond
}

// CHECK-LABEL: hw.module @print_proc
hw.module @print_proc(in %val: i8) {
  %lit = sim.fmt.literal "p="
  %hex = sim.fmt.hex %val, isUpper false : i8
  %nl = sim.fmt.literal "\0A"
  %msg = sim.fmt.concat (%lit, %hex, %nl)
  %fd = hw.constant 123 : i32
  sv.initial {
    // CHECK: sv.initial {
    // CHECK: sv.fwrite %{{.+}}, "p=%x\0A"(%val) : i8
    // CHECK: sv.fwrite %{{.+}}, "p=%x\0A"(%val) : i8
    // CHECK: sv.fflush fd %{{.+}}
    sim.proc.print %msg
    sim.proc.print %msg to %fd
    sim.proc.fflush %fd
  }
}
