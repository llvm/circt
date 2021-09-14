// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  func @foo(%arg0 : index, %arg1 : index) {
    // CHECK:      %0 = addi %arg0, %arg1 {hls.dir = #hls.pipeline<{II = 2 : i64, style = 2 : i64}>} : index
    // CHECK-NEXT: %1 = addi %arg0, %arg1 {hls.dir = #hls.function_instantiate<{variable = "abc"}>} : index
    %0 = addi %arg0, %arg1 {hls.dir = #hls.pipeline<{II = 2, style = 2}>} : index
    %1 = addi %arg0, %arg1 {hls.dir = #hls.function_instantiate<{variable = "abc"}>} : index
    return
  }
}
