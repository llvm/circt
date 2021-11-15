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

handshake.func @invalid_instance_op(%arg0 : i32, %arg1 : none, %ctrl : none) -> none {
  // expected-error @+1 {{'handshake.instance' op last operand must be a control (none-typed) operand.}}
  handshake.instance @foo(%arg1, %arg0) : (none, i32) -> ()
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

// expected-error @+1 {{'handshake.func' op attribute 'argNames' has 2 entries but is expected to have 3.}}
handshake.func @invalid_num_argnames(%a : i32, %b : i32, %c : none) -> none attributes {argNames = ["a", "b"]} {
  handshake.return %c : none
}

// -----

// expected-error @+1 {{'handshake.func' op expected all entries in attribute 'argNames' to be strings.}}
handshake.func @invalid_type_argnames(%a : i32, %b : none) -> none attributes {argNames = ["a", 2 : i32]} {
  handshake.return %b : none
}

// -----

// expected-error @+1 {{'handshake.func' op attribute 'resNames' has 1 entries but is expected to have 2.}}
handshake.func @invalid_num_resnames(%a : i32, %b : i32, %c : none) -> (i32, none) attributes {resNames = ["a"]} {
  handshake.return %a, %c : i32, none
}

// -----

// expected-error @+1 {{'handshake.func' op expected all entries in attribute 'resNames' to be strings.}}
handshake.func @invalid_type_resnames(%a : i32, %b : none) -> none attributes {resNames = [2 : i32]} {
  handshake.return %b : none
}

// -----

// expected-error @+1 {{'handshake.func' op argument 0 has multiple uses.}}
handshake.func @invalid_mutliple_use_arg(%a : i32, %ctrl : none) -> (i32, none) {
  %0 = arith.addi %a, %a : i32
  handshake.return %0, %ctrl : i32, none
}

// -----

handshake.func @invalid_mutliple_use_nonhandshake_op(%a : i32, %b : i32, %c : i32, %d : i32, %ctrl : none) -> (i32, none) {
  // expected-error @+1 {{'arith.addi' op result 0 has multiple uses.}}
  %0 = arith.addi %a, %b : i32
  %1 = arith.addi %0, %c : i32
  %2 = arith.addi %0, %d : i32
  handshake.return %2, %ctrl : i32, none
}

// -----

handshake.func @invalid_multiple_use_handshake_op(%a : i32, %ctrl : none) -> (i32, none) {
  // expected-error @+1 {{'handshake.fork' op failed to verify that all results should have exactly one use}}
  %0:2 = "handshake.fork"(%a) {control = false} : (i32) -> (i32, i32)
  %1 = arith.addi %0#0, %0#0 : i32
  handshake.return %1, %ctrl : i32, none
}
