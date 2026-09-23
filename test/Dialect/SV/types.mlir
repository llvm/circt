// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: module
module {

// CHECK-LABEL: func @netType(%arg0: !sv.net<i42>)
func.func @netType(%arg0: !sv.net<i42>) {
 return
}


// CHECK-LABEL: func @varType(%arg0: !sv.var<i42>)
func.func @varType(%arg0: !sv.var<i42>) {
 return
}
}

