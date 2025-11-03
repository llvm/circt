firrtl.circuit "InferVectorPort" {
  firrtl.module @InferVectorPort(in %in: !firrtl.vector<uint<4>, 2>, out %out: !firrtl.vector<uint, 2>) {
    // CHECK: firrtl.connect %out, %in : !firrtl.vector<uint<4>, 2>
    firrtl.connect %out, %in : !firrtl.vector<uint, 2>, !firrtl.vector<uint<4>, 2>
  }
}