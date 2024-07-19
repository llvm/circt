// RUN: circt-opt %s | circt-opt | FileCheck %s

// Test Overview:
//   * 0 observed signals, no time, successor without arguments
//   * 0 observed signals, with time, sucessor with arguments
//   * 2 observed signals, no time, successor with arguments
//   * 2 observed signals, with time, successor with arguments

// CHECK-LABEL: @check_wait_0
hw.module @check_wait_0 () {
  // CHECK-NEXT: llhd.process
  llhd.process {
    // CHECK: llhd.wait ^[[BB:.*]]
    "llhd.wait"() [^bb1] {operandSegmentSizes=array<i32: 0,0,0>} : () -> ()
    // CHECK-NEXT: ^[[BB]]
  ^bb1:
    llhd.halt
  }
}

// CHECK-LABEL: @check_wait_1
hw.module @check_wait_1 () {
  // CHECK-NEXT: llhd.process
  llhd.process {
    // CHECK-NEXT: %[[TIME:.*]] = llhd.constant_time
    %time = llhd.constant_time #llhd.time<0ns, 0d, 0e>
    // CHECK-NEXT: llhd.wait for %[[TIME]], ^[[BB:.*]](%[[TIME]] : !llhd.time)
    "llhd.wait"(%time, %time) [^bb1] {operandSegmentSizes=array<i32: 0,1,1>} : (!llhd.time, !llhd.time) -> ()
    // CHECK-NEXT: ^[[BB]](%[[T:.*]]: !llhd.time):
  ^bb1(%t: !llhd.time):
    llhd.halt
  }
}

// CHECK: @check_wait_2(inout %[[ARG0:.*]] : i64, inout %[[ARG1:.*]] : i1) {
hw.module @check_wait_2 (inout %arg0 : i64, inout %arg1 : i1) {
  // CHECK-NEXT: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.wait (%[[ARG0]], %[[ARG1]] : !hw.inout<i64>, !hw.inout<i1>), ^[[BB:.*]](%[[ARG1]] : !hw.inout<i1>)
    "llhd.wait"(%arg0, %arg1, %arg1) [^bb1] {operandSegmentSizes=array<i32: 2,0,1>} : (!hw.inout<i64>, !hw.inout<i1>, !hw.inout<i1>) -> ()
    // CHECK: ^[[BB]](%[[A:.*]]: !hw.inout<i1>):
  ^bb1(%a: !hw.inout<i1>):
    llhd.halt
  }
}

// CHECK: hw.module @check_wait_3(inout %[[ARG0:.*]] : i64, inout %[[ARG1:.*]] : i1) {
hw.module @check_wait_3 (inout %arg0 : i64, inout %arg1 : i1) {
  // CHECK-NEXT: llhd.process
  llhd.process {
    // CHECK-NEXT: %[[TIME:.*]] = llhd.constant_time
    %time = llhd.constant_time #llhd.time<0ns, 0d, 0e>
    // CHECK-NEXT: llhd.wait for %[[TIME]], (%[[ARG0]], %[[ARG1]] : !hw.inout<i64>, !hw.inout<i1>), ^[[BB:.*]](%[[ARG1]], %[[ARG0]] : !hw.inout<i1>, !hw.inout<i64>)
    "llhd.wait"(%arg0, %arg1, %time, %arg1, %arg0) [^bb1] {operandSegmentSizes=array<i32: 2,1,2>} : (!hw.inout<i64>, !hw.inout<i1>, !llhd.time, !hw.inout<i1>, !hw.inout<i64>) -> ()
    // CHECK: ^[[BB]](%[[A:.*]]: !hw.inout<i1>, %[[B:.*]]: !hw.inout<i64>):
  ^bb1(%a: !hw.inout<i1>, %b: !hw.inout<i64>):
    llhd.halt
  }
}
