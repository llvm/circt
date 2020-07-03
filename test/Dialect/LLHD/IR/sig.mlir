// RUN: circt-opt %s -split-input-file -verify-diagnostics | circt-opt | FileCheck %s

llhd.entity @check_sig_inst () -> () {
    // CHECK: %[[CI1:.*]] = llhd.const
    %cI1 = llhd.const 0 : i1
    // CHECK-NEXT: %{{.*}} = llhd.sig "sigI1" %[[CI1]] : i1
    %sigI1 = "llhd.sig"(%cI1) {name = "sigI1"} : (i1) -> !llhd.sig<i1>
    // CHECK-NEXT: %[[CI64:.*]] = llhd.const
    %cI64 = llhd.const 0 : i64
    // CHECK-NEXT: %{{.*}} = llhd.sig "sigI64" %[[CI64]] : i64
    %sigI64 = "llhd.sig"(%cI64) {name = "sigI64"} : (i64) -> !llhd.sig<i64>
}

// CHECK-LABEL: check_prb
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>
// CHECK-SAME: %[[SI64:.*]]: !llhd.sig<i64>
func @check_prb(%sigI1 : !llhd.sig<i1>, %sigI64 : !llhd.sig<i64>) {
    // CHECK: %{{.*}} = llhd.prb %[[SI1]] : !llhd.sig<i1>
    %0 = "llhd.prb"(%sigI1) {} : (!llhd.sig<i1>) -> i1
    // CHECK-NEXT: %{{.*}} = llhd.prb %[[SI64]] : !llhd.sig<i64>
    %1 = "llhd.prb"(%sigI64) {} : (!llhd.sig<i64>) -> i64

    return
}

// CHECK-LABEL: check_drv
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>
// CHECK-SAME: %[[SI64:.*]]: !llhd.sig<i64>
// CHECK-SAME: %[[CI1:.*]]: i1
// CHECK-SAME: %[[CI64:.*]]: i64
// CHECK-SAME: %[[TIME:.*]]: !llhd.time
func @check_drv(%sigI1 : !llhd.sig<i1>, %sigI64 : !llhd.sig<i64>, %cI1 : i1, %cI64 : i64, %t : !llhd.time) {
    // CHECK-NEXT: llhd.drv %[[SI1]], %[[CI1]] after %[[TIME]] : !llhd.sig<i1>
    "llhd.drv"(%sigI1, %cI1, %t) {} : (!llhd.sig<i1>, i1, !llhd.time) -> ()
    // CHECK-NEXT: llhd.drv %[[SI64]], %[[CI64]] after %[[TIME]] : !llhd.sig<i64>
    "llhd.drv" (%sigI64, %cI64, %t) {} : (!llhd.sig<i64>, i64, !llhd.time) -> ()
    // CHECK-NEXT: llhd.drv %[[SI64]], %[[CI64]] after %[[TIME]] if %[[CI1]] : !llhd.sig<i64>
    "llhd.drv" (%sigI64, %cI64, %t, %cI1) {} : (!llhd.sig<i64>, i64, !llhd.time, i1) -> ()

    return
}

// -----

// expected-error @+3 {{failed to verify that type of 'init' and underlying type of 'signal' have to match.}}
llhd.entity @check_illegal_sig () -> () {
    %cI1 = llhd.const 0 : i1
    %sig1 = "llhd.sig"(%cI1) {name="foo"} : (i1) -> !llhd.sig<i32>
}

// -----

// expected-error @+2 {{failed to verify that type of 'result' and underlying type of 'signal' have to match.}}
llhd.entity @check_illegal_prb (%sig : !llhd.sig<i1>) -> () {
    %prb = "llhd.prb"(%sig) {} : (!llhd.sig<i1>) -> i32
}

// -----

// expected-error @+4 {{failed to verify that type of 'value' and underlying type of 'signal' have to match.}}
llhd.entity @check_illegal_drv (%sig : !llhd.sig<i1>) -> () {
    %c = llhd.const 0 : i32
    %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    "llhd.drv"(%sig, %c, %time) {} : (!llhd.sig<i1>, i32, !llhd.time) -> ()
}

// -----

// expected-error @+4 {{Redefinition of signal named 'sigI1'!}}
llhd.entity @check_unique_sig_names () -> () {
    %cI1 = llhd.const 0 : i1
    %sig1 = llhd.sig "sigI1" %cI1 : i1
    %sig2 = llhd.sig "sigI1" %cI1 : i1
}
