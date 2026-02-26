// RUN: circt-verilog --ir-moore %s | FileCheck %s

// CHECK-LABEL: moore.module private @PortsUnconnected(in %a : !moore.l1, out b : !moore.l1, in %c : !moore.ref<l1>, in %d : !moore.ref<l1>) {
// CHECK:         %b = moore.net wire : <l1>
// CHECK:         %c_0 = moore.assigned_variable name "c" %1 : l1
// CHECK:         %d_1 = moore.assigned_variable name "d" %2 : l1
// CHECK:         %0 = moore.read %b : <l1>
// CHECK:         %1 = moore.read %c : <l1>
// CHECK:         %2 = moore.read %d : <l1>
// CHECK:         moore.output %0 : !moore.l1
// CHECK:       }
module PortsUnconnected(
  input a,
  output b,
  inout logic c,
  ref logic d
);
endmodule

// CHECK-LABEL: moore.module @PortsTop() {
// CHECK:         %a = moore.net wire : <l1>
// CHECK:         %0 = moore.read %a : <l1>
// CHECK:         %c = moore.net wire : <l1>
// CHECK:         %d = moore.variable : <l1>
// CHECK:         %p4.b = moore.instance "p4" @PortsUnconnected(a: %0: !moore.l1, c: %c: !moore.ref<l1>, d: %d: !moore.ref<l1>) -> (b: !moore.l1)
// CHECK:         moore.output
// CHECK:       }
module PortsTop;
  PortsUnconnected p4(
    .a(), // Unconnected input
    .b(), // Unconnected output
    .c(), // Unconnected inout
    .d()  // Unconnected ref
  );
endmodule
