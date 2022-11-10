// RUN: circt-translate --import-verilog %s | FileCheck %s

// CHECK-LABEL: moore.module @TestForeach
// CHECK:  [[t0:%.+]] = moore.constant 1 : !moore.int
// CHECK:  [[t2:%.+]] = moore.constant 2 : !moore.int
// CHECK:  [[t3:%.+]] = moore.constant 0 : !moore.int
// CHECK:  [[t4:%.+]] = scf.while ([[arg0:%.+]] = [[t3]]) : (!moore.int) -> !moore.int {
// CHECK:    [[t7:%.+]] = moore.lt [[t3]], [[t2]] : !moore.int -> !moore.bit
// CHECK:    [[t8:%.+]] = moore.conversion [[t7]] : !moore.bit -> i1
// CHECK:    scf.condition([[t8]]) [[arg0]] : !moore.int
// CHECK:  } do {
// CHECK:  ^bb0([[arg0]]: !moore.int):
// CHECK:    [[t7:%.+]] = moore.constant 0 : !moore.int
// CHECK:    [[t8:%.+]] = moore.constant 3 : !moore.int
// CHECK:    [[t9:%.+]] = moore.constant 0 : !moore.int
// CHECK:    [[t10:%.+]] = scf.while ([[arg1:%.+]] = [[t9]]) : (!moore.int) -> !moore.int {
// CHECK:      [[t12:%.+]] = moore.lt [[t9]], [[t8]] : !moore.int -> !moore.bit
// CHECK:      [[t13:%.+]] = moore.conversion [[t12]] : !moore.bit -> i1
// CHECK:      scf.condition([[t13]]) [[arg1]] : !moore.int
// CHECK:    } do {
// CHECK:    ^bb0([[arg1]]: !moore.int):
// CHECK:      moore.blocking_assign %x, %y : !moore.bit
// CHECK:      [[t14:%.+]] = moore.add [[arg1]], [[t0]] : !moore.int
// CHECK:      scf.yield [[t14]] : !moore.int
// CHECK:    }
// CHECK:    [[t11:%.+]] = moore.add [[arg0]], [[t0]] : !moore.int
// CHECK:    scf.yield [[t11]] : !moore.int
// CHECK:  }
// CHECK:}

module TestForeach;
  bit array[3][4][4][4];
  bit x, y;
  initial begin
    foreach (array[i, ,m,]) begin
      x = y;
    end
  end
endmodule
