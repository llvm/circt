// RUN: circt-opt %s | FileCheck %s
// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @test() {
func @test() {
  // Basic interface smoke test

  sv.interface @myinterface {
    sv.interface.signal "data" : i32
    sv.interface.modport "input_port" [{direction = "input", signal = @data}]
    sv.interface.modport "output_port" [{direction = "output", signal = @data}]
  }

  // CHECK-NEXT: sv.interface @myinterface {
  // CHECK-NEXT:   sv.interface.signal "data" : i32
  // CHECK-NEXT:   sv.interface.modport "input_port" [{direction = "input", signal = @data}]
  // CHECK-NEXT:   sv.interface.modport "output_port" [{direction = "output", signal = @data}]
  // CHECK-NEXT: }

  // Hanshake-like interface smoke test

  sv.interface @handshake_example {
    sv.interface.signal "data" : i32
    sv.interface.signal "valid" : i1
    sv.interface.signal "ready" : i1
    sv.interface.modport "dataflow_in" [{direction = "input", signal = @data}, {direction = "input", signal = @valid}, {direction = "output", signal = @ready}]
    sv.interface.modport "dataflow_out" [{direction = "output", signal = @data}, {direction = "output", signal = @valid}, {direction = "input", signal = @ready}]
  }

  // CHECK-NEXT: sv.interface @handshake_example {
  // CHECK-NEXT:   sv.interface.signal "data" : i32
  // CHECK-NEXT:   sv.interface.signal "valid" : i1
  // CHECK-NEXT:   sv.interface.signal "ready" : i1
  // CHECK-NEXT:   sv.interface.modport "dataflow_in" [{direction = "input", signal = @data}, {direction = "input", signal = @valid}, {direction = "output", signal = @ready}]
  // CHECK-NEXT:   sv.interface.modport "dataflow_out" [{direction = "output", signal = @data}, {direction = "output", signal = @valid}, {direction = "input", signal = @ready}]
  // CHECK-NEXT: }

  // CHECK-NEXT: return
  return
}
