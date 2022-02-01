// RUN: circt-opt %s --split-input-file --verify-diagnostics

handshake.func @invalid_merge_like_no_data(%arg0: i1) {
  // expected-error @+1 {{'handshake.mux' op must have at least one data operand}}
  %0 = mux %arg0 [] : i1, i32 
  return %0 : i32
}

// -----

handshake.func @invalid_merge_like_wrong_type(%arg0: i1, %arg1: i32, %arg2: i64) { // expected-note {{prior use here}}
  // expected-error @+1 {{use of value '%arg2' expects different type than prior uses: 'i32' vs 'i64'}}
  %0 = mux %arg0 [%arg1, %arg2] : i1, i32 
  return %0 : i32
}

// -----

handshake.func @invalid_mux_unsupported_select(%arg0: tensor<i1>, %arg1: i32, %arg2: i32) {
  // expected-error @+1 {{unsupported type for select operand: 'tensor<i1>'}}
  %0 = mux %arg0 [%arg1, %arg2] : tensor<i1>, i32
  return %0 : i32
}

// -----

handshake.func @invalid_mux_narrow_select(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: i32) {
  // expected-error @+1 {{select bitwidth was 1, which can mux 2 operands, but found 3 operands}}
  %0 = mux %arg0 [%arg1, %arg2, %arg3] : i1, i32
  return %0 : i32
}

// -----

handshake.func @foo(%ctrl : none) -> none{
  return %ctrl : none
}

handshake.func @invalid_instance_op(%arg0 : i32, %ctrl : none) -> none {
  // expected-error @+1 {{'handshake.instance' op last operand must be a control (none-typed) operand.}}
  instance @foo(%ctrl, %arg0) : (none, i32) -> ()
  return %ctrl : none
}

// -----

handshake.func @foo(%ctrl : none) -> none{
  return %ctrl : none
}

handshake.func @invalid_instance_op(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.instance' op must provide at least a control operand.}}
  instance @foo() : () -> ()
  return %ctrl : none
}

// -----

handshake.func @invalid_multidim_memory(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.memory' op memref must have only a single dimension.}}
  memory [ld = 0, st = 0] () {id = 0 : i32, lsq = false} : memref<10x10xi8>, () -> ()
  return %ctrl : none
}

// -----

handshake.func @invalid_dynamic_memory(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.memory' op memref dimensions for handshake.memory must be static.}}
  memory [ld = 0, st = 0] () {id = 0 : i32, lsq = false} : memref<?xi8>, () -> ()
  return %ctrl : none
}

// -----

// expected-error @+1 {{'handshake.func' op attribute 'argNames' has 2 entries but is expected to have 3.}}
handshake.func @invalid_num_argnames(%a : i32, %b : i32, %c : none) -> none attributes {argNames = ["a", "b"]} {
  return %c : none
}

// -----

// expected-error @+1 {{'handshake.func' op expected all entries in attribute 'argNames' to be strings.}}
handshake.func @invalid_type_argnames(%a : i32, %b : none) -> none attributes {argNames = ["a", 2 : i32]} {
  return %b : none
}

// -----

// expected-error @+1 {{'handshake.func' op attribute 'resNames' has 1 entries but is expected to have 2.}}
handshake.func @invalid_num_resnames(%a : i32, %b : i32, %c : none) -> (i32, none) attributes {resNames = ["a"]} {
  return %a, %c : i32, none
}

// -----

// expected-error @+1 {{'handshake.func' op expected all entries in attribute 'resNames' to be strings.}}
handshake.func @invalid_type_resnames(%a : i32, %b : none) -> none attributes {resNames = [2 : i32]} {
  return %b : none
}

// -----

handshake.func @invalid_constant_value(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.constant' op constant value type differs from operation result type.}}
  %0 = constant %ctrl {value = 1 : i31} : i32
  return %ctrl : none
}

// -----

handshake.func @invalid_buffer_init1(%arg0 : i32, %ctrl : none) -> (i32, none) {
  // expected-error @+1 {{'handshake.buffer' op expected 2 init values but got 1.}}
  %0 = buffer [2] %arg0 {initValues = [1], sequential=true} : i32
  return %0, %ctrl : i32, none
}

// -----

handshake.func @invalid_buffer_init2(%arg0 : i32, %ctrl : none) -> (i32, none) {
  // expected-error @+1 {{'handshake.buffer' op only sequential buffers are allowed to have initial values.}}
  %0 = buffer [1] %arg0 {initValues = [1], sequential=false} : i32
  return %0, %ctrl : i32, none
}
