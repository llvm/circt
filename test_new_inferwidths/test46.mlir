firrtl.circuit "StringAndUInt" { 
  // https://github.com/llvm/circt/issues/5983
  // Just check propassign doesn't cause an error.
  firrtl.module @StringAndUInt(in %x: !firrtl.uint<5>,
                                      out %y: !firrtl.uint,
                                      out %s: !firrtl.string) {
    firrtl.connect %y, %x : !firrtl.uint, !firrtl.uint<5>
    %0 = firrtl.string "test"
    firrtl.propassign %s, %0 : !firrtl.string
  }
}