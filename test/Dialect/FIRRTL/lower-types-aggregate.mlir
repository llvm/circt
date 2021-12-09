// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-types{preserve-aggregate=true})' %s | FileCheck %s

firrtl.circuit "VectorSimple" {
  // CHECK-LABEL: firrtl.module @VectorSimple(in %source: !firrtl.vector<uint<1>, 2>, out %sink: !firrtl.vector<uint<1>, 2>)
  firrtl.module @VectorSimple(in %source: !firrtl.vector<uint<1>, 2>, out %sink: !firrtl.vector<uint<1>, 2>) {
    // CHECK-NEXT: firrtl.connect %sink, %source
    firrtl.connect %sink, %source : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
} // CIRCUIT
