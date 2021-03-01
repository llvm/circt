// RUN: circt-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @if_dead_condition(%arg0: i1) {
// CHECK-NEXT:    sv.always posedge %arg0  {
// CHECK-NEXT:      sv.fwrite "Reachable1"
// CHECK-NEXT:      sv.fwrite "Reachable2"
// CHECK-NEXT:      sv.fwrite "Reachable3"
// CHECK-NEXT:      sv.fwrite "Reachable4"
// CHECK-NEXT:    }
// CHECK-NEXT:    return 
// CHECK-NEXT:  }

func @if_dead_condition(%arg0: i1) {
  sv.always posedge %arg0 {
    %true = rtl.constant true
    %false = rtl.constant false

    sv.if %true {}

    sv.if %false {
        sv.fwrite "Unreachable0"
    }

    sv.if %true {
      sv.fwrite "Reachable1"
    }

    sv.if %true {
      sv.fwrite "Reachable2"
    } else {
      sv.fwrite "Unreachable2"
    } 

    sv.if %false {
      sv.fwrite "Unreachable3"
    } else {
      sv.fwrite "Reachable3"
    } 

    sv.if %false {
      sv.fwrite "Unreachable4"
    } else {
      sv.fwrite "Reachable4"
    } 
  }
  return
}