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
