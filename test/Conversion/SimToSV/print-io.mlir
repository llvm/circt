// RUN: circt-opt --sim-proceduralize --lower-hw-to-sv --lower-sim-to-sv %s | FileCheck %s

sv.macro.decl @SYNTHESIS
sv.macro.decl @PRINTF_COND_
sv.func private @"__circt_lib_logging::FileDescriptor::get"(in %name : !hw.string, out fd : i32 {sv.func.explicitly_returned}) attributes {verilogName = "__circt_lib_logging::FileDescriptor::get"}

// CHECK-LABEL: hw.module @print_io
// CHECK:          [[CLK:%.+]] = seq.from_clock %clk
// CHECK-NEXT:     sv.always posedge [[CLK]] {
// CHECK-NEXT:       %[[TIME:.*]] = sv.system.time : i64
// CHECK-NEXT:       %[[NAME:.*]] = sv.sformatf "out_%0d.log"(%val) : i8
// CHECK-NEXT:       %[[FD:.*]] = sv.func.call.procedural @"__circt_lib_logging::FileDescriptor::get"(%[[NAME]]) : (!hw.string) -> i32
// CHECK-NEXT:       sv.if %cond {
// CHECK:              sv.fwrite %{{.+}}, "%0t"(%[[TIME]]) : i64
// CHECK:              sv.fwrite %{{.+}}, "v=%x\0A"(%val) : i8
// CHECK:              sv.fwrite %[[FD]], "v=%x\0A"(%val) : i8
// CHECK-NEXT:       }
// CHECK-NEXT:     }
hw.module @print_io(in %clk: !seq.clock, in %cond: i1, in %val: i8) {
  %lit = sim.fmt.literal "v="
  %hex = sim.fmt.hex %val, isUpper false : i8
  %nl = sim.fmt.literal "\0A"
  %msg = sim.fmt.concat (%lit, %hex, %nl)
  %dec = sim.fmt.dec %val signed : i8
  %t = sim.time
  %tfmt = sim.fmt.dec %t : i64
  sim.print %tfmt on %clk if %cond {usePrintfCond = true}
  %tmsg = sim.fmt.concat (%dec, %nl)
  sim.print %msg on %clk if %cond {usePrintfCond = true}

  %fd = sim.get_file "out_%0d.log"(%val) : (i8) -> i32
  sim.print %msg to %fd on %clk if %cond
}
