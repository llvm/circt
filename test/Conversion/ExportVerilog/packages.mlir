// RUN: circt-opt %s -export-verilog -o %t.mlir | FileCheck %s
// RUN: FileCheck %s --check-prefix=IR < %t.mlir
// RUN: circt-opt %t.mlir -export-verilog -o /dev/null | FileCheck %s

hw.module.extern @external() attributes {verilogName = "types"}

// Package names are global; typedefs and enum members share a local namespace.
// CHECK-LABEL: package types_0;
// CHECK: typedef logic [7:0] logic_0;
// CHECK: typedef logic_0 wordAlias;
// CHECK: typedef enum bit [0:0] {State_Idle_0, State_Busy} State;
// CHECK: typedef State stateAlias;
// CHECK: typedef logic [1:0] State_Idle;
// CHECK: endpackage
sv.package @types {
  // IR: hw.typedecl @word, "logic_0" : i8
  hw.typedecl @word, "logic" : i8
  hw.typedecl @wordAlias : !hw.typealias<@types::@word, i8>
  hw.typedecl @state, "State" : !hw.enum<Idle, Busy>
  hw.typedecl @stateAlias : !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>
  // Even a later typedef must reserve its name before enum members are named.
  hw.typedecl @State_Idle : i2
}

// CHECK-LABEL: package other;
// CHECK: typedef logic [15:0] word;
// CHECK: typedef types_0::logic_0 imported;
// CHECK: endpackage
sv.package @other {
  hw.typedecl @word : i16
  hw.typedecl @imported : !hw.typealias<@types::@word, i8>
}

// CHECK-LABEL: module Consumer(
// CHECK: input {{ *}}types_0::logic_0 word
// CHECK: input {{ *}}other::word
// CHECK: output types_0::logic_0 result
hw.module @Consumer(
    in %clock: i1, in %word: !hw.typealias<@types::@word, i8>,
    in %otherWord: !hw.typealias<@other::@word, i16>,
    out result: !hw.typealias<@types::@word, i8>) {
  // CHECK: types_0::stateAlias state;
  %state = sv.reg : !hw.inout<!hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>>>
  %value = sv.read_inout %state : !hw.inout<!hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>>>
  %idle = hw.enum.constant Idle : !hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>>
  %busy = hw.enum.constant Busy : !hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>>
  // CHECK: case (state)
  // CHECK-NEXT: types_0::State_Idle_0:
  // CHECK-NEXT: state <= types_0::State_Busy;
  // CHECK-NEXT: default:
  // CHECK-NEXT: state <= types_0::State_Idle_0;
  sv.always posedge %clock {
    sv.case case %value : !hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>>
      case Idle: { sv.passign %state, %busy : !hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>> }
      default: { sv.passign %state, %idle : !hw.typealias<@types::@stateAlias, !hw.typealias<@types::@state, !hw.enum<Idle, Busy>>> }
  }
  hw.output %word : !hw.typealias<@types::@word, i8>
}
