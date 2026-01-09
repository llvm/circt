// RUN: circt-opt %s --arc-insert-runtime --split-input-file | FileCheck %s --check-prefixes=CHECK,NOARGS
// RUN: circt-opt %s --arc-insert-runtime='extra-args="debug;bar"' --split-input-file | FileCheck %s --check-prefixes=CHECK,RTARGS

// CHECK-DAG: llvm.func @arcRuntimeIR_allocInstance(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @arcRuntimeIR_deleteInstance(!llvm.ptr)
// CHECK-DAG: llvm.func @arcRuntimeIR_onEval(!llvm.ptr)
// CHECK-DAG: arc.runtime.model @arcRuntimeModel_counter "counter" numStateBytes 4

arc.runtime.model @noTouchy "dontTouch" numStateBytes 4

arc.model @counter io !hw.modty<input clk : i1, output o : i8> {
^bb0(%arg0: !arc.storage<4>):
    %c1_i8 = hw.constant 1 : i8
    %in_clk = arc.root_input "clk", %arg0 {offset = 0 : i32} : (!arc.storage<4>) -> !arc.state<i1>
    %0 = arc.alloc_state %arg0 {offset = 1 : i32} : (!arc.storage<4>) -> !arc.state<i8>
    %1 = arc.alloc_state %arg0 {offset = 2 : i32} : (!arc.storage<4>) -> !arc.state<i1>
    %out_o = arc.root_output "o", %arg0 {offset = 3 : i32} : (!arc.storage<4>) -> !arc.state<i8>
    %2 = arc.storage.get %arg0[0] : !arc.storage<4> -> !arc.state<i1>
    %3 = arc.state_read %2 : <i1>
    %4 = arc.storage.get %arg0[2] : !arc.storage<4> -> !arc.state<i1>
    %5 = arc.state_read %4 : <i1>
    arc.state_write %4 = %3 : <i1>
    %6 = comb.xor %5, %3 : i1
    %7 = comb.and %6, %3 : i1
    scf.if %7 {
        %11 = arc.storage.get %arg0[1] : !arc.storage<4> -> !arc.state<i8>
        %12 = arc.state_read %11 : <i8>
        %13 = comb.add %12, %c1_i8 : i8
        arc.state_write %11 = %13 : <i8>
    }
    %8 = arc.storage.get %arg0[1] : !arc.storage<4> -> !arc.state<i8>
    %9 = arc.state_read %8 : <i8>
    %10 = arc.storage.get %arg0[3] : !arc.storage<4> -> !arc.state<i8>
    arc.state_write %10 = %9 : <i8>
}

func.func @main() {
  %zero = arith.constant 0 : i1
  %one = arith.constant 1 : i1
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index

  // NOARGS-LABEL: arc.sim.instantiate @counter as %arg0 runtime @arcRuntimeModel_counter()
  // RTARGS-LABEL: arc.sim.instantiate @counter as %arg0 runtime @arcRuntimeModel_counter("debug;bar")
  arc.sim.instantiate @counter as %model {
    // CHECK: [[RTINST:%.*]] = builtin.unrealized_conversion_cast %arg0 : !arc.sim.instance<@counter> to !llvm.ptr
    scf.for %i = %lb to %ub step %step {
      // CHECK:      arc.sim.set_input
      // CHECK-NEXT: llvm.call @arcRuntimeIR_onEval([[RTINST]])
      // CHECK-NEXT: arc.sim.step
      // CHECK-NEXT: arc.sim.set_input
      // CHECK-NEXT: llvm.call @arcRuntimeIR_onEval([[RTINST]])
      // CHECK-NEXT: arc.sim.step
      // CHECK-NOT:  arcRuntimeIR_onEval
      arc.sim.set_input %model, "clk" = %one : i1, !arc.sim.instance<@counter>
      arc.sim.step %model : !arc.sim.instance<@counter>
      arc.sim.set_input %model, "clk" = %zero : i1, !arc.sim.instance<@counter>
      arc.sim.step %model : !arc.sim.instance<@counter>
    }
  }

  // NOARGS-LABEL: arc.sim.instantiate @counter as %arg0 runtime @arcRuntimeModel_counter("foo")
  // RTARGS-LABEL: arc.sim.instantiate @counter as %arg0 runtime @arcRuntimeModel_counter("foo;debug;bar")
  arc.sim.instantiate @counter as %model runtime ("foo") {
    // CHECK: [[RTINST:%.*]] = builtin.unrealized_conversion_cast %arg0 : !arc.sim.instance<@counter> to !llvm.ptr
    // CHECK-NEXT: llvm.call @arcRuntimeIR_onEval([[RTINST]])
    // CHECK-NEXT: arc.sim.step
    // CHECK-NOT:  arcRuntimeIR_onEval
    arc.sim.step %model : !arc.sim.instance<@counter>
  }

  // CHECK-LABEL: arc.sim.instantiate @counter as %arg0 runtime @noTouchy("foo")
  // CHECK-NOT: arcRuntimeIR_onEval
  arc.sim.instantiate @counter as %model runtime @noTouchy("foo") {
    arc.sim.step %model : !arc.sim.instance<@counter>
  }
  return
}


// -----

// Model without instance

// CHECK-NOT: llvm.func @arcRuntimeIR_
// CHECK: arc.runtime.model @arcRuntimeModel_counter "counter" numStateBytes 4
// CHECK-NOT: llvm.func @arcRuntimeIR_

arc.model @counter io !hw.modty<input clk : i1, output o : i8> {
^bb0(%arg0: !arc.storage<4>):
    %c1_i8 = hw.constant 1 : i8
    %in_clk = arc.root_input "clk", %arg0 {offset = 0 : i32} : (!arc.storage<4>) -> !arc.state<i1>
    %0 = arc.alloc_state %arg0 {offset = 1 : i32} : (!arc.storage<4>) -> !arc.state<i8>
    %1 = arc.alloc_state %arg0 {offset = 2 : i32} : (!arc.storage<4>) -> !arc.state<i1>
    %out_o = arc.root_output "o", %arg0 {offset = 3 : i32} : (!arc.storage<4>) -> !arc.state<i8>
    %2 = arc.storage.get %arg0[0] : !arc.storage<4> -> !arc.state<i1>
    %3 = arc.state_read %2 : <i1>
    %4 = arc.storage.get %arg0[2] : !arc.storage<4> -> !arc.state<i1>
    %5 = arc.state_read %4 : <i1>
    arc.state_write %4 = %3 : <i1>
    %6 = comb.xor %5, %3 : i1
    %7 = comb.and %6, %3 : i1
    scf.if %7 {
        %11 = arc.storage.get %arg0[1] : !arc.storage<4> -> !arc.state<i8>
        %12 = arc.state_read %11 : <i8>
        %13 = comb.add %12, %c1_i8 : i8
        arc.state_write %11 = %13 : <i8>
    }
    %8 = arc.storage.get %arg0[1] : !arc.storage<4> -> !arc.state<i8>
    %9 = arc.state_read %8 : <i8>
    %10 = arc.storage.get %arg0[3] : !arc.storage<4> -> !arc.state<i8>
    arc.state_write %10 = %9 : <i8>
}
