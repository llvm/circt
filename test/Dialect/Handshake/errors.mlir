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

// -----

handshake.func @invalid_multidim_memory() {
  // expected-error @+1 {{'handshake.memory' op memref must have only a single dimension.}}
  "handshake.memory"() {type = memref<10x10xi8>, id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32} : () -> ()
  return
}

// -----

handshake.func @invalid_dynamic_memory() {
  // expected-error @+1 {{'handshake.memory' op memref dimensions for handshake.memory must be static.}}
  "handshake.memory"() {type = memref<?xi8>, id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32} : () -> ()
  return
}

// -----

func @foo(%ctrl : none) -> none{
  return %ctrl : none
}

handshake.func @invalid_instance_op(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.instance' op symbol 'foo' is not a handshake.func operation.}}
  %0 = handshake.instance @foo(%ctrl) : (none) -> (none)
  handshake.return %ctrl : none
}

// -----

func @foo(%ctrl : none) -> none{
  return %ctrl : none
}

handshake.func @main(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.call' op symbol 'foo' must refer to a hw module.}}
  %0 = handshake.call @foo(%ctrl) : (none) -> (none)
  handshake.return %ctrl : none
}

// -----

hw.module.extern @foo(%in : !esi.channel<i32>) -> ()
handshake.func @main(%a : i32, %b : i32, %ctrl : none) -> none {
  // expected-error @+1 {{'handshake.call' op argument number mismatch; expected 'foo' to have at least 2 arguments, but 'foo' has 1 arguments.}}
  handshake.call @foo(%a, %b) : (i32, i32) -> ()
  handshake.return %ctrl : none
}

// -----

hw.module.extern @foo(%in : i32) -> ()
handshake.func @main(%a : i32, %ctrl : none) -> none {
  // expected-error @+1 {{'handshake.call' op expected ESI channel port as argument #0 to 'foo'.}}
  handshake.call @foo(%a) : (i32) -> ()
  handshake.return %ctrl : none
}

// -----

hw.module.extern @foo(%in : !esi.channel<i4>) -> ()
handshake.func @main(%a : i32, %ctrl : none) -> none {
  // expected-error @+1 {{'handshake.call' op channel type mismatch; expected 'i32' as inner type of port #0 of 'foo' but got 'i4'.}}
  handshake.call @foo(%a) : (i32) -> ()
  handshake.return %ctrl : none
}

// -----

hw.module.extern @foo() -> (out : i32)
handshake.func @main(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.call' op expected ESI channel port as result #0 to 'foo'.}}
  %0 = handshake.call @foo() : () -> (i32)
  handshake.return %ctrl : none
}
