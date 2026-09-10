// RUN: circt-opt %s -export-verilog -o %t.mlir | FileCheck %s
// RUN: circt-opt %t.mlir -export-verilog -o /dev/null | FileCheck %s

hw.module.extern @external() attributes {verilogName = "foo"}

// CHECK-LABEL: package foo_0;
// CHECK: typedef logic [31:0] logic_0;
// CHECK: typedef logic [7:0] logic_1;
// CHECK: typedef logic [1:0] State_Idle;
// CHECK: typedef enum bit [0:0] {State_Idle_0, State_Busy} State;
// CHECK: // typedef logic {{.*}}empty;
// CHECK: endpackage
sv.package @foo {
  hw.typedecl @word, "logic" : i32
  hw.typedecl @byte, "logic" : i8
  hw.typedecl @State_Idle : i2
  hw.typedecl @state, "State" : !hw.enum<Idle, Busy>
  hw.typedecl @empty : i0
}

// CHECK-LABEL: interface Signals;
// CHECK: foo_0::logic_0 value;
sv.interface @Signals {
  sv.interface.signal @value : !hw.typealias<@foo::@word, i32>
}

// CHECK-LABEL: module Consumer
// CHECK: input {{ *}}foo_0::logic_0
// CHECK: foo_0::logic_1
// CHECK: output foo_0::State
// CHECK: assign state = foo_0::State_Idle_0;
hw.module @Consumer(
    in %word: !hw.typealias<@foo::@word, i32>,
    in %byte: !hw.typealias<@foo::@byte, i8>,
    out state: !hw.typealias<@foo::@state, !hw.enum<Idle, Busy>>) {
  %idle = hw.enum.constant Idle : !hw.typealias<@foo::@state, !hw.enum<Idle, Busy>>
  hw.output %idle : !hw.typealias<@foo::@state, !hw.enum<Idle, Busy>>
}
