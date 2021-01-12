// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_store_3ins_2outs
// CHECK: %[[IN_DATA_VALID:.+]] = firrtl.subfield %arg0("valid")
// CHECK: %[[IN_DATA_READY:.+]] = firrtl.subfield %arg0("ready")
// CHECK: %[[IN_DATA_DATA:.+]] = firrtl.subfield %arg0("data")
// CHECK: %[[IN_ADDR_VALID:.+]] = firrtl.subfield %arg1("valid")
// CHECK: %[[IN_ADDR_READY:.+]] = firrtl.subfield %arg1("ready")
// CHECK: %[[IN_ADDR_DATA:.+]] = firrtl.subfield %arg1("data")
// CHECK: %[[IN_CONTROL_VALID:.+]] = firrtl.subfield %arg2("valid")
// CHECK: %[[IN_CONTROL_READY:.+]] = firrtl.subfield %arg2("ready")
// CHECK: %[[OUT_DATA_VALID:.+]] = firrtl.subfield %arg3("valid")
// CHECK: %[[OUT_DATA_READY:.+]] = firrtl.subfield %arg3("ready")
// CHECK: %[[OUT_DATA_DATA:.+]] = firrtl.subfield %arg3("data")
// CHECK: %[[OUT_ADDR_VALID:.+]] = firrtl.subfield %arg4("valid")
// CHECK: %[[OUT_ADDR_READY:.+]] = firrtl.subfield %arg4("ready")
// CHECK: %[[OUT_ADDR_DATA:.+]] = firrtl.subfield %arg4("data")

// CHECK: firrtl.connect %[[OUT_ADDR_DATA]], %[[IN_ADDR_DATA]]
// CHECK: firrtl.connect %[[OUT_DATA_DATA]], %[[IN_DATA_DATA]]

// CHECK: %[[ALL_VALID0:.+]] = firrtl.and %[[IN_DATA_VALID]], %[[IN_ADDR_VALID]]
// CHECK: %[[ALL_VALID:.+]] = firrtl.and %[[ALL_VALID0]], %[[IN_CONTROL_VALID]]

// CHECK: firrtl.connect %[[OUT_DATA_VALID]], %[[ALL_VALID]]
// CHECK: firrtl.connect %[[OUT_ADDR_VALID]], %[[ALL_VALID]]

// CHECK: %[[ALL_READY:.+]] = firrtl.and %[[OUT_DATA_READY]], %[[OUT_ADDR_READY]]

// CHECK: firrtl.connect %[[IN_DATA_READY]], %[[ALL_READY]]
// CHECK: firrtl.connect %[[IN_ADDR_READY]], %[[ALL_READY]]
// CHECK: firrtl.connect %[[IN_CONTROL_READY]], %[[ALL_READY]]

// CHECK-LABEL: firrtl.module @main
handshake.func @main(%arg0: i8, %arg1: index, %arg2: none, ...) -> (i8, index, none) {
  // CHECK: %0 = firrtl.instance @handshake_store_3ins_2outs
  %0:2 = "handshake.store"(%arg0, %arg1, %arg2) : (i8, index, none) -> (i8, index)

  handshake.return %0#0, %0#1, %arg2 : i8, index, none
}
