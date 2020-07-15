// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics | circt-opt | circt-opt | FileCheck %s

llhd.entity @check_sig_inst () -> () {
  // CHECK: %[[CI1:.*]] = llhd.const
  %cI1 = llhd.const 0 : i1
  // CHECK-NEXT: %{{.*}} = llhd.sig "sigI1" %[[CI1]] : i1
  %sigI1 = llhd.sig "sigI1" %cI1 : i1
  // CHECK-NEXT: %[[CI64:.*]] = llhd.const
  %cI64 = llhd.const 0 : i64
  // CHECK-NEXT: %{{.*}} = llhd.sig "sigI64" %[[CI64]] : i64
  %sigI64 = llhd.sig "sigI64" %cI64 : i64

  // CHECK-NEXT: %[[TUP:.*]] = llhd.tuple
  %tup = llhd.tuple %cI1, %cI64 : tuple<i1, i64>
  // CHECK-NEXT: %{{.*}} = llhd.sig "sigTup" %[[TUP]] : tuple<i1, i64>
  %sigTup = llhd.sig "sigTup" %tup : tuple<i1, i64>

  // CHECK-NEXT: %[[VEC:.*]] = llhd.vec
  %vec = llhd.vec %cI1, %cI1 : vector<2xi1>
  // CHECK-NEXT: %{{.*}} = llhd.sig "sigVec" %[[VEC]] : vector<2xi1>
  %sigVec = llhd.sig "sigVec" %vec : vector<2xi1>
}

// CHECK-LABEL: check_prb
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>
// CHECK-SAME: %[[SI64:.*]]: !llhd.sig<i64>
// CHECK-SAME: %[[VEC:.*]]: !llhd.sig<vector<3xi8>>
// CHECK-SAME: %[[TUP:.*]]: !llhd.sig<tuple<i1, i2, i4>>
func @check_prb(%sigI1 : !llhd.sig<i1>, %sigI64 : !llhd.sig<i64>, %sigVec : !llhd.sig<vector<3xi8>>, %sigTup : !llhd.sig<tuple<i1, i2, i4>>) {
  // CHECK: %{{.*}} = llhd.prb %[[SI1]] : !llhd.sig<i1>
  %0 = llhd.prb %sigI1 : !llhd.sig<i1>
  // CHECK-NEXT: %{{.*}} = llhd.prb %[[SI64]] : !llhd.sig<i64>
  %1 = llhd.prb %sigI64 : !llhd.sig<i64>
  // CHECK-NEXT: %{{.*}} = llhd.prb %[[VEC]] : !llhd.sig<vector<3xi8>>
  %2 = llhd.prb %sigVec : !llhd.sig<vector<3xi8>>
  // CHECK-NEXT: %{{.*}} = llhd.prb %[[TUP]] : !llhd.sig<tuple<i1, i2, i4>>
  %3 = llhd.prb %sigTup : !llhd.sig<tuple<i1, i2, i4>>

  return
}

// CHECK-LABEL: check_drv
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>
// CHECK-SAME: %[[SI64:.*]]: !llhd.sig<i64>
// CHECK-SAME: %[[CI1:.*]]: i1
// CHECK-SAME: %[[CI64:.*]]: i64
// CHECK-SAME: %[[TIME:.*]]: !llhd.time
// CHECK-SAME: %[[SVEC:.*]]: !llhd.sig<vector<3xi8>>
// CHECK-SAME: %[[STUP:.*]]: !llhd.sig<tuple<i1, i2, i4>>
// CHECK-SAME: %[[VEC:.*]]: vector<3xi8>
// CHECK-SAME: %[[TUP:.*]]: tuple<i1, i2, i4>
func @check_drv(%sigI1 : !llhd.sig<i1>, %sigI64 : !llhd.sig<i64>, %cI1 : i1, %cI64 : i64, %t : !llhd.time, %sigVec : !llhd.sig<vector<3xi8>>, %sigTup : !llhd.sig<tuple<i1, i2, i4>>, %vec : vector<3xi8>, %tup : tuple<i1, i2, i4>) {
  // CHECK-NEXT: llhd.drv %[[SI1]], %[[CI1]] after %[[TIME]] : !llhd.sig<i1>
  llhd.drv %sigI1, %cI1 after %t : !llhd.sig<i1>
  // CHECK-NEXT: llhd.drv %[[SI64]], %[[CI64]] after %[[TIME]] : !llhd.sig<i64>
  llhd.drv %sigI64, %cI64 after %t : !llhd.sig<i64>
  // CHECK-NEXT: llhd.drv %[[SI64]], %[[CI64]] after %[[TIME]] if %[[CI1]] : !llhd.sig<i64>
  llhd.drv %sigI64, %cI64 after %t if %cI1 : !llhd.sig<i64>
  // CHECK-NEXT: llhd.drv %[[SVEC]], %[[VEC]] after %[[TIME]] : !llhd.sig<vector<3xi8>>
  llhd.drv %sigVec, %vec after %t : !llhd.sig<vector<3xi8>>
  // CHECK-NEXT: llhd.drv %[[STUP]], %[[TUP]] after %[[TIME]] : !llhd.sig<tuple<i1, i2, i4>>
  llhd.drv %sigTup, %tup after %t : !llhd.sig<tuple<i1, i2, i4>>

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
