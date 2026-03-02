// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test that DPI-C open array types (byte[], int[]) are converted to
// Moore open array types (!moore.open_uarray<T>).

// CHECK: func.func private @process_data(!moore.open_uarray<i8>)
import "DPI-C" function void process_data(input byte data[]);

// CHECK: func.func private @read_write(!moore.open_uarray<i8>, !moore.ref<open_uarray<i8>>)
import "DPI-C" function void read_write(input byte wd[], output byte rd[]);

// CHECK: func.func private @int_array_fn(!moore.open_uarray<i32>)
import "DPI-C" function void int_array_fn(input int data[]);

// CHECK: func.func private @packed_bits_fn(!moore.open_array<i1>)
import "DPI-C" function void packed_bits_fn(input bit [] data);

// CHECK-LABEL: moore.module @OpenArrayCallTest
module OpenArrayCallTest(input logic clock);
  byte mydata[];
  byte result[];
  int idata[];
  bit [7:0] pdata;

  // CHECK: func.call @process_data
  // CHECK: func.call @read_write
  // CHECK: func.call @int_array_fn
  // CHECK: func.call @packed_bits_fn
  always @(posedge clock) begin
    process_data(mydata);
    read_write(mydata, result);
    int_array_fn(idata);
    packed_bits_fn(pdata);
  end
endmodule
