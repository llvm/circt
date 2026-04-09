// RUN: circt-verilog --ir-moore %s | circt-opt --moore-create-vtables --convert-moore-to-core --convert-to-llvm --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.mlir.global internal constant @"testClassVirtualInt::vtable"()
// CHECK-DAG: llvm.mlir.global internal constant @"testDerivedVirtualInt::vtable"()

// CHECK-LABEL: llvm.func @"testClassVirtualInt::adjust"(
// CHECK:      llvm.getelementptr
// CHECK:      llvm.load
// CHECK:      comb.add
// CHECK:      llvm.return

// CHECK-LABEL: llvm.func @"testDerivedVirtualInt::adjust"(
// CHECK:      llvm.getelementptr
// CHECK:      llvm.load
// CHECK:      comb.icmp slt
// CHECK:      cf.cond_br
// CHECK:      llvm.getelementptr
// CHECK:      llvm.load
// CHECK:      comb.add
// CHECK:      llvm.return
// CHECK:      llvm.load
// CHECK:      comb.sub
// CHECK:      llvm.return
// CHECK-NOT:  unrealized_conversion_cast
// CHECK-NOT:  llhd.prb

// CHECK-LABEL: hw.module @top()
// CHECK:      %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NOT:  llhd.sig %[[NULL]] : !llvm.ptr
// CHECK:      llhd.process {
// CHECK-NOT:  llhd.prb
// CHECK:      llvm.getelementptr %[[NULL]]
// CHECK:      llvm.load
// CHECK:      llvm.getelementptr
// CHECK:      llvm.load
// CHECK:      llvm.call %{{.*}}(%[[NULL]], %{{.*}}) : !llvm.ptr, (!llvm.ptr, i32) -> i32
// CHECK:      llhd.drv

class testClassVirtualInt;
  int bias;
  virtual function int adjust(int x);
    return x + bias;
  endfunction
endclass

class testDerivedVirtualInt extends testClassVirtualInt;
  int delta;
  virtual function int adjust(int x);
    if (x < delta)
      return x + bias;
    return x - delta;
  endfunction
endclass

module top;
  testClassVirtualInt t;
  int y;
  initial begin
    y = t.adjust(3);
  end
endmodule
