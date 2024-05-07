// RUN: circt-translate --export-smtlib %s | FileCheck %s
// RUN: circt-translate --export-smtlib --smtlibexport-inline-single-use-values %s | FileCheck %s --check-prefix=CHECK-INLINED

smt.solver () : () -> () {

  %true = smt.constant true

  %8 = smt.exists {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    %5 = smt.implies %4, %true
    smt.yield %5 : !smt.bool
  } patterns {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %4: !smt.bool
  }, {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    %5 = smt.implies %4, %true
    smt.yield %5 : !smt.bool
  }
  smt.assert %8
}

// how the output should look like 
// (assert (let ((tmp (exists ((tmp_0 Int) (tmp_1 Int))
// ( ! (let ((tmp_2 (= tmp_0 tmp_1)))
//                (let ((tmp_3 (=> tmp_2 true)))
//                tmp_3))
//                :pattern ((let ((tmp_4 (= tmp_0 tmp_1))) 
//                tmp_4)
//                (let ((tmp_5 (= tmp_0 tmp_1))) (let ((tmp_6 (=> tmp_5 true))) tmp_6))
//                )))))
//         tmp))
// (reset)
// (check-sat)