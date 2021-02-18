// RUN: circt-opt %s --split-input-file --canonicalize --cse | FileCheck %s

// CHECK-LABEL: cmerge_with_control_sunk
handshake.func @cmerge_with_control_sunk(%arg0: none, %arg1: none, %arg2: none) -> (none, none) {
  // CHECK: "handshake.merge"(%arg0, %arg1)
  // CHECK-NOT: "handshake.control_merge"
  // CHECK-NOT: "handshake.sink"
  %result, %index = "handshake.control_merge"(%arg0, %arg1) {control = true} : (none, none) -> (none, index)
  "handshake.sink"(%index) : (index) -> ()
  handshake.return %result, %arg2 : none, none
}
