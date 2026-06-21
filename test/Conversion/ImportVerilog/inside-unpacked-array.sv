// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s | FileCheck %s

// CHECK-LABEL: moore.module @InsideUnpackedArray
module InsideUnpackedArray
    (input int needle,
     output bit found);
  int values[3];

  initial begin
    found = needle inside {values};
  end
endmodule

// CHECK: moore.extract {{.*}} from 0 : uarray<3 x i32> -> i32
// CHECK: moore.wildcard_eq
// CHECK: moore.extract {{.*}} from 1 : uarray<3 x i32> -> i32
// CHECK: moore.wildcard_eq
// CHECK: moore.or
// CHECK: moore.extract {{.*}} from 2 : uarray<3 x i32> -> i32
// CHECK: moore.wildcard_eq
// CHECK: moore.or
