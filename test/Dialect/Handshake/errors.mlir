// RUN: circt-opt %s --split-input-file --verify-diagnostics

handshake.func @invalid_merge_like_no_data(%arg0: i1) {
  // expected-error @+1 {{'handshake.mux' op must have at least one data operand}}
  %0 = "handshake.mux"(%arg0) : (i1) -> (i32)
  return %0 : i32
}

// -----

handshake.func @invalid_merge_like_wrong_type(%arg0: i1, %arg1: i32, %arg2: i64) {
  // expected-error @+1 {{'handshake.mux' op operand has type 'i64', but result has type 'i32'}}
  %0 = "handshake.mux"(%arg0, %arg1, %arg2) : (i1, i32, i64) -> (i32)
  return %0 : i32
}

// -----

handshake.func @invalid_mux_single_operand(%arg0: i1, %arg1: i32) {
  // expected-error @+1 {{need at least two inputs to mux}}
  %0 = "handshake.mux"(%arg0, %arg1) : (i1, i32) -> (i32)
  return %0 : i32
}

// -----

handshake.func @invalid_mux_unsupported_select(%arg0: tensor<i1>, %arg1: i32, %arg2: i32) {
  // expected-error @+1 {{unsupported type for select operand: 'tensor<i1>'}}
  %0 = "handshake.mux"(%arg0, %arg1, %arg2) : (tensor<i1>, i32, i32) -> (i32)
  return %0 : i32
}

// -----

handshake.func @invalid_mux_narrow_select(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: i32) {
  // expected-error @+1 {{select bitwidth was 1, which can mux 2 operands, but found 3 operands}}
  %0 = "handshake.mux"(%arg0, %arg1, %arg2, %arg3) : (i1, i32, i32, i32) -> (i32)
  return %0 : i32
}

// -----

handshake.func @foo(%ctrl : none) -> none{
  handshake.return %ctrl : none
}

handshake.func @invalid_instance_op(%arg0 : i32, %ctrl : none) -> none {
  // expected-error @+1 {{'handshake.instance' op last operand must be a control (none-typed) operand.}}
  handshake.instance @foo(%ctrl, %arg0) : (none, i32) -> ()
  handshake.return %ctrl : none
}

// -----

handshake.func @foo(%ctrl : none) -> none{
  handshake.return %ctrl : none
}

handshake.func @invalid_instance_op(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.instance' op must provide at least a control operand.}}
  handshake.instance @foo() : () -> ()
  handshake.return %ctrl : none
}
