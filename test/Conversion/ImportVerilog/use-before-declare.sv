// RUN: not circt-verilog --ir-moore %s 2>&1 | FileCheck %s --check-prefix=NOALLOW
// RUN: circt-verilog --ir-moore --allow-use-before-declare %s | FileCheck %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// NOALLOW: identifier 'a' used before its declaration

// CHECK-LABEL: moore.module @ForwardProceduralUse()
module ForwardProceduralUse;
  // CHECK-DAG: %a = moore.variable : <i32>
  // CHECK-DAG: %b = moore.variable : <i32>
  // CHECK: moore.procedure initial {
  // CHECK:   moore.read %b : <i32>
  // CHECK:   moore.blocking_assign %a, {{.*}} : i32
  // CHECK: }
  initial a = b;

  int a;
  int b;
endmodule

// CHECK-LABEL: moore.module @ForwardVariableInitializers(
module ForwardVariableInitializers(output int out);
  // CHECK-DAG: %[[A:[[:alnum:]_]+]] = moore.variable %[[AINIT:[[:alnum:]_]+]] : <i32>
  // CHECK-DAG: %[[B:[[:alnum:]_]+]] = moore.variable {{.*}} : <i32>
  // CHECK-DAG: %[[AREAD:[[:alnum:]_]+]] = moore.read %[[A]] : <i32>
  // CHECK-DAG: %[[BREAD:[[:alnum:]_]+]] = moore.read %[[B]] : <i32>
  // CHECK-DAG: %[[AINIT]] = moore.add %[[BREAD]]
  // CHECK: moore.output %[[AREAD]] : !moore.i32
  int a = b + 1;
  int b = 41;
  assign out = a;
endmodule

// CHECK-LABEL: moore.module @ForwardNetDeclarationAssign(
module ForwardNetDeclarationAssign(input logic in, output logic out);
  // CHECK: %tmp = moore.assigned_variable %late : l1
  // CHECK: %late = moore.assigned_variable %in : l1
  // CHECK: moore.output %tmp : !moore.l1
  assign out = tmp;
  wire tmp = late;
  wire late = in;
endmodule

interface ForwardVif(input logic clk);
  logic sig;
endinterface

// CHECK-LABEL: moore.module @ForwardVirtualInterfaceUse(
module ForwardVirtualInterfaceUse(input logic clk);
  ForwardVif bus(clk);

  // The virtual interface declaration is intentionally after the use. In
  // use-before-declare mode, the predeclaration pass still has to register
  // virtual interface members so `vif.sig` resolves while lowering the
  // procedure.
  initial begin
    // CHECK: moore.struct_extract
    vif.sig = value;
  end

  bit value;
  virtual ForwardVif vif = bus;
endmodule
