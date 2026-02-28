// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK: [[FALSE:%.+]] = arith.constant false
// CHECK: [[TRUE:%.+]] = arith.constant true
// CHECK: scf.for [[I:%.+]] = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}, [[VIOLATED:%.+]] = {{%.+}})
// CHECK: [[IGNOREUNTIL:%.+]] = arith.constant 3
// CHECK: [[CMP:%.+]] = arith.cmpi ult, [[I]], [[IGNOREUNTIL]]
// CHECK: [[NEWVIOLATED:%.+]] = scf.if [[CMP]]
// CHECK:     scf.yield [[VIOLATED]]
// CHECK: } else {
// CHECK:     [[CHECK:%.+]] = smt.check sat {
// CHECK:     smt.yield [[TRUE]]
// CHECK:     } unknown {
// CHECK:     smt.yield [[TRUE]]
// CHECK:     } unsat {
// CHECK:     smt.yield [[FALSE]]
// CHECK:     } -> i1
// CHECK:     [[OR:%.+]] = arith.ori [[CHECK]], [[VIOLATED]]
// CHECK:     scf.yield [[OR]]
// CHECK: }
// CHECK: func.call @bmc_loop()
// CHECK: [[FUNCDECL:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK: scf.yield [[FUNCDECL]], [[NEWVIOLATED]]


func.func @test_bmc() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 0 initial_values [] attributes {ignore_asserts_until = 3 : i64}
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%arg0: i32):
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %arg0 : i32
  }
  func.return %bmc : i1
}
