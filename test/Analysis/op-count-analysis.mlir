// RUN: circt-opt -test-op-count-analysis %s 2>&1 | FileCheck %s

// CHECK: builtin.module: 1
// CHECK:  with 0 operands: 1
// CHECK: comb.add: 4
// CHECK:  with 1 operands: 1
// CHECK:  with 2 operands: 1
// CHECK:  with 3 operands: 2
// CHECK: comb.icmp: 1
// CHECK:  with 2 operands: 1
// CHECK: comb.xor: 1
// CHECK:  with 1 operands: 1
// CHECK: hw.module: 1
// CHECK:  with 0 operands: 1
// CHECK: hw.output: 1
// CHECK:  with 0 operands: 1
// CHECK: scf.if: 1
// CHECK:  with 1 operands: 1
// CHECK: scf.yield: 2
// CHECK:  with 1 operands: 2

module {
  hw.module @bar(in %in1: i8, in %in2: i8, in %in3: i8) {
    %add2 = comb.add %in1, %in2 : i8
    %add3 = comb.add %in1, %in2, %in3 : i8
    %add3again = comb.add %in1, %in3, %in3 : i8
    %gt = comb.icmp ult %in1, %in2 : i8
    %x = scf.if %gt -> (i8) {
        %add1 = comb.add %in1 : i8
        scf.yield %add1 : i8
    } else {
        %xor = comb.xor %add2 : i8
        scf.yield %xor : i8
    }
  }
}
