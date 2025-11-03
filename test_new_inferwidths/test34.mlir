firrtl.circuit "InferVectorFancy" {
  firrtl.module @InferVectorFancy(in %in : !firrtl.uint<4>) {
    // CHECK: firrtl.wire : !firrtl.vector<uint<4>, 10>
    %wv = firrtl.wire : !firrtl.vector<uint, 10>
    %wv_5 = firrtl.subindex %wv[5] : !firrtl.vector<uint, 10>
    firrtl.connect %wv_5, %in : !firrtl.uint, !firrtl.uint<4>

    // CHECK: firrtl.wire : !firrtl.bundle<a: uint<4>>
    %wb = firrtl.wire : !firrtl.bundle<a: uint>
    %wb_a = firrtl.subfield %wb[a] : !firrtl.bundle<a: uint>

    %wv_2 = firrtl.subindex %wv[2] : !firrtl.vector<uint, 10>
    firrtl.connect %wb_a, %wv_2 : !firrtl.uint, !firrtl.uint
  }
}