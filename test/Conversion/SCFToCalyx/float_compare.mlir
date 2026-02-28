// RUN: circt-opt %s --lower-scf-to-calyx -canonicalize -split-input-file | FileCheck %s

// Test floating point OEQ

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_1.left, %std_and_1.right, %std_and_1.out = calyx.std_and @std_and_1 : i1, i1, i1
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:      calyx.group @bb0_0 {
// CHECK-DAG:        calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:        calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:        calyx.assign %std_compareFN_0.signaling = %false : i1
// CHECK-DAG:        calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:        calyx.assign %compare_port_0_reg.in = %std_compareFN_0.eq : i1
// CHECK-DAG:        calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:        calyx.assign %std_not_0.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:        calyx.assign %unordered_port_0_reg.in = %std_not_0.out : i1
// CHECK-DAG:        calyx.assign %std_and_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:        calyx.assign %std_and_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:        calyx.assign %std_and_1.left = %compare_port_0_reg.done : i1
// CHECK-DAG:        calyx.assign %std_and_1.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:        calyx.assign %cmpf_0_reg.in = %std_and_0.out : i1
// CHECK-DAG:        calyx.assign %cmpf_0_reg.write_en = %std_and_1.out : i1
// CHECK-DAG:        %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:        calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:        calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:      }
// CHECK:         calyx.group @ret_assign_0 {
// CHECK-DAG:           calyx.assign %ret_arg0_reg.in = %cmpf_0_reg.out : i1
// CHECK-DAG:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:         }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf oeq, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point OGT

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_1.left, %std_and_1.right, %std_and_1.out = calyx.std_and @std_and_1 : i1, i1, i1
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:       calyx.group @bb0_0 {
// CHECK-DAG:         calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.signaling = %true : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.in = %std_compareFN_0.gt : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %std_not_0.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.in = %std_not_0.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_1.left = %compare_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %std_and_1.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.in = %std_and_0.out : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.write_en = %std_and_1.out : i1
// CHECK-DAG:         %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:         calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:         calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:       }
// CHECK:         calyx.group @ret_assign_0 {
// CHECK-DAG:           calyx.assign %ret_arg0_reg.in = %cmpf_0_reg.out : i1
// CHECK-DAG:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:         }
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf ogt, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point OGE

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_1.left, %std_and_1.right, %std_and_1.out = calyx.std_and @std_and_1 : i1, i1, i1
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:       calyx.group @bb0_0 {
// CHECK-DAG:         calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.signaling = %true : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %std_not_0.in = %std_compareFN_0.lt : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.in = %std_not_0.out : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %std_not_1.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.in = %std_not_1.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_1.left = %compare_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %std_and_1.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.in = %std_and_0.out : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.write_en = %std_and_1.out : i1
// CHECK-DAG:         %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:         calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:         calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:       }
// CHECK:         calyx.group @ret_assign_0 {
// CHECK-DAG:           calyx.assign %ret_arg0_reg.in = %cmpf_0_reg.out : i1
// CHECK-DAG:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:         }
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf oge, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point OLT

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_1.left, %std_and_1.right, %std_and_1.out = calyx.std_and @std_and_1 : i1, i1, i1
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:       calyx.group @bb0_0 {
// CHECK-DAG:         calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.signaling = %true : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.in = %std_compareFN_0.lt : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %std_not_0.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.in = %std_not_0.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_1.left = %compare_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %std_and_1.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.in = %std_and_0.out : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.write_en = %std_and_1.out : i1
// CHECK-DAG:         %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:         calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:         calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:       }
// CHECK:         calyx.group @ret_assign_0 {
// CHECK-DAG:           calyx.assign %ret_arg0_reg.in = %cmpf_0_reg.out : i1
// CHECK-DAG:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:         }
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf olt, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point OLE

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_1.left, %std_and_1.right, %std_and_1.out = calyx.std_and @std_and_1 : i1, i1, i1
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:        calyx.group @bb0_0 {
// CHECK-DAG:          calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:          calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:          calyx.assign %std_compareFN_0.signaling = %true : i1
// CHECK-DAG:          calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:          calyx.assign %std_not_0.in = %std_compareFN_0.gt : i1
// CHECK-DAG:          calyx.assign %compare_port_0_reg.in = %std_not_0.out : i1
// CHECK-DAG:          calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:          calyx.assign %std_not_1.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:          calyx.assign %unordered_port_0_reg.in = %std_not_1.out : i1
// CHECK-DAG:          calyx.assign %std_and_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:          calyx.assign %std_and_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:          calyx.assign %std_and_1.left = %compare_port_0_reg.done : i1
// CHECK-DAG:          calyx.assign %std_and_1.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:          calyx.assign %cmpf_0_reg.in = %std_and_0.out : i1
// CHECK-DAG:          calyx.assign %cmpf_0_reg.write_en = %std_and_1.out : i1
// CHECK-DAG:          %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:          calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:          calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:        }
// CHECK:         calyx.group @ret_assign_0 {
// CHECK-DAG:           calyx.assign %ret_arg0_reg.in = %cmpf_0_reg.out : i1
// CHECK-DAG:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:         }
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf ole, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point ONE

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_1.left, %std_and_1.right, %std_and_1.out = calyx.std_and @std_and_1 : i1, i1, i1
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:       calyx.group @bb0_0 {
// CHECK-DAG:         calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.signaling = %false : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %std_not_0.in = %std_compareFN_0.eq : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.in = %std_not_0.out : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %std_not_1.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.in = %std_not_1.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_1.left = %compare_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %std_and_1.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.in = %std_and_0.out : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.write_en = %std_and_1.out : i1
// CHECK-DAG:         %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:         calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:         calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:       }
// CHECK:         calyx.group @ret_assign_0 {
// CHECK-DAG:           calyx.assign %ret_arg0_reg.in = %cmpf_0_reg.out : i1
// CHECK-DAG:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:         }
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf one, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point ORD

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:       calyx.group @bb0_0 {
// CHECK-DAG:         calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.signaling = %false : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %std_not_0.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.in = %std_not_0.out : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.in = %unordered_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.write_en = %unordered_port_0_reg.done : i1
// CHECK-DAG:         %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:         calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:         calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf ord, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point UEQ

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %std_or_0.left, %std_or_0.right, %std_or_0.out = calyx.std_or @std_or_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:      calyx.group @bb0_0 {
// CHECK-DAG:        calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:        calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:        calyx.assign %std_compareFN_0.signaling = %false : i1
// CHECK-DAG:        calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:        calyx.assign %compare_port_0_reg.in = %std_compareFN_0.eq : i1
// CHECK-DAG:        calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:        calyx.assign %unordered_port_0_reg.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:        calyx.assign %std_or_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:        calyx.assign %std_or_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:        calyx.assign %std_and_0.left = %compare_port_0_reg.done : i1
// CHECK-DAG:        calyx.assign %std_and_0.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:        calyx.assign %cmpf_0_reg.in = %std_or_0.out : i1
// CHECK-DAG:        calyx.assign %cmpf_0_reg.write_en = %std_and_0.out : i1
// CHECK-DAG:        %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:        calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:        calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:      }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf ueq, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point UGT

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %std_or_0.left, %std_or_0.right, %std_or_0.out = calyx.std_or @std_or_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:       calyx.group @bb0_0 {
// CHECK-DAG:         calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.signaling = %true : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.in = %std_compareFN_0.gt : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:         calyx.assign %std_or_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_or_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.left = %compare_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %std_and_0.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.in = %std_or_0.out : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.write_en = %std_and_0.out : i1
// CHECK-DAG:         %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:         calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:         calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf ugt, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point UGE

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %std_or_0.left, %std_or_0.right, %std_or_0.out = calyx.std_or @std_or_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:       calyx.group @bb0_0 {
// CHECK-DAG:         calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.signaling = %true : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %std_not_0.in = %std_compareFN_0.lt : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.in = %std_not_0.out : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:         calyx.assign %std_or_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_or_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.left = %compare_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %std_and_0.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.in = %std_or_0.out : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.write_en = %std_and_0.out : i1
// CHECK-DAG:         %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:         calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:         calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf uge, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point ULT

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %std_or_0.left, %std_or_0.right, %std_or_0.out = calyx.std_or @std_or_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:      calyx.group @bb0_0 {
// CHECK-DAG:        calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:        calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:        calyx.assign %std_compareFN_0.signaling = %true : i1
// CHECK-DAG:        calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:        calyx.assign %compare_port_0_reg.in = %std_compareFN_0.lt : i1
// CHECK-DAG:        calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:        calyx.assign %unordered_port_0_reg.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:        calyx.assign %std_or_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:        calyx.assign %std_or_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:        calyx.assign %std_and_0.left = %compare_port_0_reg.done : i1
// CHECK-DAG:        calyx.assign %std_and_0.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:        calyx.assign %cmpf_0_reg.in = %std_or_0.out : i1
// CHECK-DAG:        calyx.assign %cmpf_0_reg.write_en = %std_and_0.out : i1
// CHECK-DAG:        %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:        calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:        calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:      }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf ult, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point ULE

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %std_or_0.left, %std_or_0.right, %std_or_0.out = calyx.std_or @std_or_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:       calyx.group @bb0_0 {
// CHECK-DAG:         calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.signaling = %true : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %std_not_0.in = %std_compareFN_0.gt : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.in = %std_not_0.out : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:         calyx.assign %std_or_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_or_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.left = %compare_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %std_and_0.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.in = %std_or_0.out : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.write_en = %std_and_0.out : i1
// CHECK-DAG:         %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:         calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:         calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf ule, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point UNE

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:       %std_or_0.left, %std_or_0.right, %std_or_0.out = calyx.std_or @std_or_0 : i1, i1, i1
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:       calyx.group @bb0_0 {
// CHECK-DAG:         calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.signaling = %false : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %std_not_0.in = %std_compareFN_0.eq : i1
// CHECK-DAG:         calyx.assign %compare_port_0_reg.in = %std_not_0.out : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:         calyx.assign %std_or_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_or_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %std_and_0.left = %compare_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %std_and_0.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.in = %std_or_0.out : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.write_en = %std_and_0.out : i1
// CHECK-DAG:         %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:         calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:         calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf une, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point UNO

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:       calyx.group @bb0_0 {
// CHECK-DAG:         calyx.assign %std_compareFN_0.left = %in0 : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.right = %cst : i32
// CHECK-DAG:         calyx.assign %std_compareFN_0.signaling = %false : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:         calyx.assign %unordered_port_0_reg.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.in = %unordered_port_0_reg.out : i1
// CHECK-DAG:         calyx.assign %cmpf_0_reg.write_en = %unordered_port_0_reg.done : i1
// CHECK-DAG:         %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:         calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:         calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf uno, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point AlwaysTrue

// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:         calyx.group @ret_assign_0 {
// CHECK-DAG:           calyx.assign %ret_arg0_reg.in = %true : i1
// CHECK-DAG:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:         }
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf true, %arg0, %0 : f32

    return %1 : i1
  }
}

// -----

// Test floating point AlwaysFalse


// CHECK:   module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i1, %done: i1 {done}) {
// CHECK-DAG:       %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:       %true = hw.constant true
// CHECK-DAG:       %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i1, i1, i1, i1, i1, i1
// CHECK:       calyx.wires {
// CHECK-DAG:         calyx.assign %out0 = %ret_arg0_reg.out : i1
// CHECK:         calyx.group @ret_assign_0 {
// CHECK-DAG:           calyx.assign %ret_arg0_reg.in = %false : i1
// CHECK-DAG:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:         }
// CHECK-DAG:       }

module {
  func.func @main(%arg0 : f32) -> i1 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.cmpf false, %arg0, %0 : f32

    return %1 : i1
  }
}
