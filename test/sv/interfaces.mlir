// RUN: circt-opt %s | FileCheck %s
// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @test() {
func @test() {
  // Basic interface smoke test

  sv.interface @myinterface {
    sv.interface.signal @data : i32
    sv.interface.modport @input_port ("input" @data)
    sv.interface.modport @output_port ("output" @data)
  }

  // CHECK-NEXT: sv.interface @myinterface {
  // CHECK-NEXT:   sv.interface.signal @data : i32
  // CHECK-NEXT:   sv.interface.modport @input_port ("input" @data)
  // CHECK-NEXT:   sv.interface.modport @output_port ("output" @data)
  // CHECK-NEXT: }

  // Hanshake-like interface smoke test

  sv.interface @handshake_example {
    sv.interface.signal @data : i32
    sv.interface.signal @valid : i1
    sv.interface.signal @ready : i1
    sv.interface.modport @dataflow_in ("input" @data, "input" @valid, "output" @ready)
    sv.interface.modport @dataflow_out ("output" @data, "output" @valid, "input" @ready)
  }

  // CHECK-NEXT: sv.interface @handshake_example {
  // CHECK-NEXT:   sv.interface.signal @data : i32
  // CHECK-NEXT:   sv.interface.signal @valid : i1
  // CHECK-NEXT:   sv.interface.signal @ready : i1
  // CHECK-NEXT:   sv.interface.modport @dataflow_in ("input" @data, "input" @valid, "output" @ready)
  // CHECK-NEXT:   sv.interface.modport @dataflow_out ("output" @data, "output" @valid, "input" @ready)
  // CHECK-NEXT: }

  // CHECK-NEXT: return
  return
}
