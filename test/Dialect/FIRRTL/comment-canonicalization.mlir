// RUN: circt-opt -canonicalize='top-down=true region-simplify=aggressive' %s | FileCheck %s

firrtl.circuit "CommentCanonicalization" {
  // CHECK-LABEL: firrtl.module @CommentCanonicalization
  firrtl.module @CommentCanonicalization(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, in %i: !firrtl.uint<1>, out %node_out: !firrtl.uint<1>, out %wire_out: !firrtl.uint<1>, out %single_set_reg_out: !firrtl.uint<1>, out %zero_out: !firrtl.uint<1>, out %one_out: !firrtl.uint<1>, out %const_out: !firrtl.uint<1>, out %reset_mux_out: !firrtl.uint<1>) {
    // A live node carrying a comment must retain its declaration.
    // CHECK: %named_node = firrtl.node %i
    // CHECK-SAME: comment = "node comment"
    %named_node = firrtl.node %i {comment = "node comment"} : !firrtl.uint<1>
    firrtl.matchingconnect %node_out, %named_node : !firrtl.uint<1>

    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    // Single-connect forwarding must not erase a live declaration carrying
    // a comment.
    // CHECK: %single_set_wire = firrtl.wire
    // CHECK-SAME: comment = "single set wire comment"
    %single_set_wire = firrtl.wire droppable_name {comment = "single set wire comment"} : !firrtl.uint<1>
    firrtl.matchingconnect %single_set_wire, %i : !firrtl.uint<1>
    firrtl.matchingconnect %wire_out, %single_set_wire : !firrtl.uint<1>

    // CHECK: %single_set_reg = firrtl.reg %clock
    // CHECK-SAME: comment = "single set reg comment"
    %single_set_reg = firrtl.reg droppable_name %clock {comment = "single set reg comment"} : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %single_set_reg, %c0_ui1 : !firrtl.uint<1>
    firrtl.matchingconnect %single_set_reg_out, %single_set_reg : !firrtl.uint<1>

    // A reset-less identity rewrite must preserve the declaration comment.
    // CHECK: %zero_reset = firrtl.reg %clock
    // CHECK-SAME: comment = "zero reset comment"
    %zero_reset = firrtl.regreset %clock, %c0_ui1, %i {comment = "zero reset comment"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.matchingconnect %zero_out, %zero_reset : !firrtl.uint<1>

    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    // An always-reset register is replaced by a node carrier.
    // CHECK: %one_reset = firrtl.node %i
    // CHECK-SAME: comment = "one reset comment"
    %one_reset = firrtl.regreset %clock, %c1_ui1, %i {comment = "one reset comment"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.matchingconnect %one_out, %one_reset : !firrtl.uint<1>

    // Do not replace a live register carrying a comment with a constant.
    // CHECK: %constant_reg = firrtl.reg %clock
    // CHECK-SAME: comment = "constant reg comment"
    %constant_reg = firrtl.reg %clock {comment = "constant reg comment"} : !firrtl.clock, !firrtl.uint<1>
    %constant_mux = firrtl.mux(%cond, %c0_ui1, %constant_reg) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %constant_reg, %constant_mux : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.matchingconnect %const_out, %constant_reg : !firrtl.uint<1>

    // Do not fold a live reset register into its reset constant when its comment
    // would have no declaration carrier.
    // CHECK: %reset_mux_reg = firrtl.regreset %clock, %reset, %c0_ui1
    // CHECK-SAME: comment = "reset mux comment"
    %reset_mux_reg = firrtl.regreset %clock, %reset, %c0_ui1 {comment = "reset mux comment"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    %reset_mux = firrtl.mux(%reset, %c0_ui1, %reset_mux_reg) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %reset_mux_reg, %reset_mux : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.matchingconnect %reset_mux_out, %reset_mux_reg : !firrtl.uint<1>
  }

  // Comments must not keep declarations alive when all uses are confined to
  // their next-state feedback loop.
  // CHECK-LABEL: firrtl.module private @DeadComments
  // CHECK-NOT: %dead_node =
  // CHECK-NOT: %dead_wire =
  // CHECK-NOT: %dead_reg =
  // CHECK-NOT: %dead_regreset =
  // CHECK-NOT: dead node comment
  // CHECK-NOT: dead wire comment
  // CHECK-NOT: dead reg comment
  // CHECK-NOT: dead regreset comment
  firrtl.module private @DeadComments(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, in %i: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    %dead_node = firrtl.node %i {comment = "dead node comment"} : !firrtl.uint<1>

    %dead_wire = firrtl.wire {comment = "dead wire comment"} : !firrtl.uint<1>
    firrtl.matchingconnect %dead_wire, %i : !firrtl.uint<1>

    %dead_reg = firrtl.reg %clock {comment = "dead reg comment"} : !firrtl.clock, !firrtl.uint<1>
    %dead_reg_mux = firrtl.mux(%cond, %c0_ui1, %dead_reg) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %dead_reg, %dead_reg_mux : !firrtl.uint<1>, !firrtl.uint<1>

    %dead_regreset = firrtl.regreset %clock, %reset, %c0_ui1 {comment = "dead regreset comment"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    %dead_regreset_mux = firrtl.mux(%reset, %c0_ui1, %dead_regreset) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %dead_regreset, %dead_regreset_mux : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @AttachComment
  firrtl.module @AttachComment(in %a: !firrtl.analog<1>) {
    // An attached wire is semantically live and has no replacement declaration
    // carrier, so its declaration comment blocks removal.
    // CHECK: %attached_wire = firrtl.wire
    // CHECK-SAME: comment = "attached wire comment"
    %attached_wire = firrtl.wire {comment = "attached wire comment"} : !firrtl.analog<1>
    // CHECK: firrtl.attach %attached_wire, %a
    firrtl.attach %attached_wire, %a : !firrtl.analog<1>, !firrtl.analog<1>
  }
}
