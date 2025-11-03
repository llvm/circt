firrtl.circuit "AttachMany" {
  firrtl.module @AttachMany(
    in %a0: !firrtl.analog<8>,
    in %a1: !firrtl.analog,
    in %a2: !firrtl.analog<8>,
    in %a3: !firrtl.analog) {
    firrtl.attach %a0, %a1, %a2, %a3 : !firrtl.analog<8>, !firrtl.analog, !firrtl.analog<8>, !firrtl.analog
  }
}