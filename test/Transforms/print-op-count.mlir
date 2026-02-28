// RUN: circt-opt -print-op-count=emission-format=readable-sorted %s 2>&1 | FileCheck %s

// CHECK: - name: builtin.module
// CHECK:   count: 1
// CHECK: - name: comb.add
// CHECK:   count: 4
// CHECK:     - operands: 1
// CHECK:       count: 1
// CHECK:     - operands: 2
// CHECK:       count: 1
// CHECK:     - operands: 3
// CHECK:       count: 2
// CHECK: - name: comb.icmp
// CHECK:   count: 1
// CHECK: - name: comb.xor
// CHECK:   count: 1
// CHECK: - name: hw.module
// CHECK:   count: 1
// CHECK: - name: hw.output
// CHECK:   count: 1
// CHECK: - name: scf.if
// CHECK:   count: 1
// CHECK: - name: scf.yield
// CHECK:   count: 2

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
