// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK: ssp.instance "no properties" of "Problem" {
// CHECK:   library {  
// CHECK:     operator_type @NoProps
// CHECK:   }
// CHECK:   graph {
// CHECK:     %[[op_0:.*]] = operation<> @Op0()
// CHECK:     operation<>(%[[op_0]])
// CHECK:     operation<>(@Op0)
// CHECK:     operation<>(%[[op_0]], @Op0)
// CHECK:   }
// CHECK: }
ssp.instance "no properties" of "Problem" {
  library {
    operator_type @NoProps
  }
  graph {
    %0 = operation<> @Op0()
    operation<>(%0)
    operation<>(@Op0)
    operation<>(%0, @Op0)
  }
}
