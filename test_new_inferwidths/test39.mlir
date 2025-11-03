firrtl.circuit "AttachTwo" {
  firrtl.module @AttachTwo(in %a0: !firrtl.analog<8>, in %a1: !firrtl.analog) {
    firrtl.attach %a0, %a1 : !firrtl.analog<8>, !firrtl.analog
  }
}