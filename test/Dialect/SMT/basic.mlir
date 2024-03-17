// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @types
// CHECK-SAME:  (%{{.*}}: !smt.bool, %{{.*}}: !smt.bv<32>)
func.func @types(%arg0: !smt.bool, %arg1: !smt.bv<32>) {
  return
}

func.func @core(%in: i8) {
  // CHECK: %a = smt.declare_const "a" {smt.some_attr} : !smt.bool
  %a = smt.declare_const "a" {smt.some_attr} : !smt.bool
  // CHECK: smt.declare_const {smt.some_attr} : !smt.bv<32>
  %b = smt.declare_const {smt.some_attr} : !smt.bv<32>

  // CHECK: smt.assert %a {smt.some_attr}
  smt.assert %a {smt.some_attr}

  // CHECK: %{{.*}} = smt.solver(%{{.*}}) {smt.some_attr} : (i8) -> (i8, i32) {
  // CHECK: ^bb0(%{{.*}}: i8)
  // CHECK:   %{{.*}} = smt.check {smt.some_attr} sat {
  // CHECK:     smt.yield %{{.*}} : i32
  // CHECK:   } unknown {
  // CHECK:     smt.yield %{{.*}} : i32
  // CHECK:   } unsat {
  // CHECK:     smt.yield %{{.*}} : i32
  // CHECK:   } -> i32
  // CHECK:   smt.yield %{{.*}}, %{{.*}} : i8, i32
  // CHECK: }
  %0:2 = smt.solver(%in) {smt.some_attr} : (i8) -> (i8, i32) {
  ^bb0(%arg0: i8):
    %1 = smt.check {smt.some_attr} sat {
      %c1_i32 = arith.constant 1 : i32
      smt.yield %c1_i32 : i32
    } unknown {
      %c0_i32 = arith.constant 0 : i32
      smt.yield %c0_i32 : i32
    } unsat {
      %c-1_i32 = arith.constant -1 : i32
      smt.yield %c-1_i32 : i32
    } -> i32
    smt.yield %arg0, %1 : i8, i32
  }

  // CHECK: smt.solver() : () -> () {
  // CHECK-NEXT: }
  smt.solver() : () -> () { }

  //      CHECK: smt.check sat {
  // CHECK-NEXT: } unknown {
  // CHECK-NEXT: } unsat {
  // CHECK-NEXT: }
  smt.check sat { } unknown { } unsat { }

  // CHECK: %{{.*}} = smt.eq %{{.*}}, %{{.*}} {smt.some_attr} : !smt.bv<32>
  %1 = smt.eq %b, %b {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.distinct %{{.*}}, %{{.*}} {smt.some_attr} : !smt.bv<32>
  %2 = smt.distinct %b, %b {smt.some_attr} : !smt.bv<32>

  // CHECK: %{{.*}} = smt.eq %{{.*}}, %{{.*}}, %{{.*}} : !smt.bool
  %3 = smt.eq %a, %a, %a : !smt.bool
  // CHECK: %{{.*}} = smt.distinct %{{.*}}, %{{.*}}, %{{.*}} : !smt.bool
  %4 = smt.distinct %a, %a, %a : !smt.bool

  // CHECK: %{{.*}} = smt.ite %{{.*}}, %{{.*}}, %{{.*}} {smt.some_attr} : !smt.bv<32>
  %5 = smt.ite %a, %b, %b {smt.some_attr} : !smt.bv<32>

  // CEHCK: %{{.*}} = smt.not %{{.*}} {smt.some_attr}
  %6 = smt.not %a {smt.some_attr}
  // CHECK: %{{.*}} = smt.and %{{.*}}, %{{.*}}, %{{.*}} {smt.some_attr}
  %7 = smt.and %a, %a, %a {smt.some_attr}
  // CHECK: %{{.*}} = smt.or %{{.*}}, %{{.*}}, %{{.*}} {smt.some_attr}
  %8 = smt.or %a, %a, %a {smt.some_attr}
  // CHECK: %{{.*}} = smt.xor %{{.*}}, %{{.*}}, %{{.*}} {smt.some_attr}
  %9 = smt.xor %a, %a, %a {smt.some_attr}
  // CHECK: %{{.*}} = smt.implies %{{.*}}, %{{.*}} {smt.some_attr}
  %10 = smt.implies %a, %a {smt.some_attr}

  return
}
