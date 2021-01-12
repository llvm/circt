// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file -verify-diagnostics %s

// expected-error @+1 {{failed to legalize operation 'handshake.func'}}
handshake.func @main(%arg0: index, %arg1: none, ...) -> none {
  %0:2 = "handshake.memory"(%addressResults) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 0 : i32, type = memref<10xi8>} : (index) -> (i8, none)
  %1:2 = "handshake.fork"(%arg1) {control = true} : (none) -> (none, none)
  %2 = "handshake.join"(%1#1, %0#1) {control = true} : (none, none) -> none
  // expected-error @+1 {{unsupported operation type}}
  %3, %addressResults = "handshake.load"(%arg0, %0#0, %1#0) : (index, i8, none) -> (i8, index)
  "handshake.sink"(%3) : (i8) -> ()
  handshake.return %2 : none
}
