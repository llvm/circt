// RUN: arcilator --run --jit-entry=AdderTest %s | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK: simulation started
// CHECK: simulation passed

verif.simulation @AdderTest {} {
^bb0(%clock: !seq.clock, %init: i1):
  // Count the first 9001 simulation cycles.
  %c0_i19 = hw.constant 0 : i19
  %c1_i19 = hw.constant 1 : i19
  %c9001_i19 = hw.constant 9001 : i19
  %count = seq.compreg %0, %clock reset %init, %c0_i19 : i19
  %done = comb.icmp eq %count, %c9001_i19 : i19
  %0 = comb.add %count, %c1_i19 : i19

  // Generate inputs to the adder.
  %1, %2 = sim.func.dpi.call @generateAdderInputs(%count) : (i19) -> (i42, i42)
  %3 = hw.instance "dut" @Adder(a: %1: i42, b: %2: i42) -> (c: i42)

  // Check results and track failures.
  %4 = comb.add %1, %2 : i42
  %5 = comb.icmp eq %3, %4 : i42
  %true = hw.constant true
  %success = seq.compreg %6, %clock reset %init, %true : i1
  %6 = comb.and %success, %5 : i1
  %7 = comb.xor %5, %true : i1

  // Print some statistics about the test.
  sim.func.dpi.call @printStart() clock %clock enable %init : () -> ()
  sim.func.dpi.call @printFailure(%1, %2, %3, %4) clock %clock enable %7 : (i42, i42, i42, i42) -> ()
  sim.func.dpi.call @printDone(%success) clock %clock enable %done : (i1) -> ()

  verif.yield %done, %success : i1, i1
}

hw.module private @Adder(in %a: i42, in %b: i42, out c: i42) {
  %0 = comb.add %a, %b : i42
  hw.output %0 : i42
}

sim.func.dpi @generateAdderInputs(in %idx: i19, out a: i42, out b: i42) attributes {verilogName = "generateAdderInputs.impl"}
func.func private @generateAdderInputs.impl(%arg0: i19, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  %c4224496855063_i42 = arith.constant 4224496855063 : i42
  %c1_i42 = arith.constant 1 : i42
  %0 = arith.extui %arg0 : i19 to i42
  %1 = arith.addi %0, %c1_i42 : i42
  %a = arith.muli %0, %c4224496855063_i42 : i42
  %b = arith.muli %1, %c4224496855063_i42 : i42
  llvm.store %a, %arg1 : i42, !llvm.ptr
  llvm.store %b, %arg2 : i42, !llvm.ptr
  return
}

sim.func.dpi @printStart() attributes {verilogName = "printStart.impl"}
sim.func.dpi @printDone(in %success: i1) attributes {verilogName = "printDone.impl"}
sim.func.dpi @printFailure(in %a: i42, in %b: i42, in %c_act: i42, in %c_exp: i42) attributes {verilogName = "printFailure.impl"}

func.func private @printStart.impl() {
  %0 = llvm.mlir.addressof @str.start : !llvm.ptr
  call @puts(%0) : (!llvm.ptr) -> ()
  return
}

func.func private @printFailure.impl(%a: i42, %b: i42, %c_act: i42, %c_exp: i42) {
  %0 = llvm.mlir.addressof @str.mismatch : !llvm.ptr
  call @puts(%0) : (!llvm.ptr) -> ()
  arc.sim.emit "a", %a : i42
  arc.sim.emit "b", %b : i42
  arc.sim.emit "c_act", %c_act : i42
  arc.sim.emit "c_exp", %c_exp : i42
  return
}

func.func private @printDone.impl(%success: i1) {
  %0 = llvm.mlir.addressof @str.pass : !llvm.ptr
  %1 = llvm.mlir.addressof @str.fail : !llvm.ptr
  %2 = llvm.select %success, %0, %1 : i1, !llvm.ptr
  call @puts(%2) : (!llvm.ptr) -> ()
  return
}

func.func private @puts(%arg0: !llvm.ptr)
llvm.mlir.global constant @str.start("simulation started\00")
llvm.mlir.global constant @str.pass("simulation passed\00")
llvm.mlir.global constant @str.fail("simulation failed\00")
llvm.mlir.global constant @str.mismatch("----- MISMATCH -----\00")
