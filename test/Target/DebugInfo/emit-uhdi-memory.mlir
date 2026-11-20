// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 | FileCheck %s

// A memory variable has `storage` `mem` and an unpacked array type.

// CHECK:      "types":[{"kind":"uint","width":8},{"elem":0,"kind":"vector","size":16}],
// CHECK-SAME: "original":{"Mem":{
// CHECK-SAME: "Memory":{"type":1,
// CHECK-SAME: "storage":"mem"
// CHECK-SAME: "vars":{"Memory":"Memory"}
hw.module @Mem() {
  %Memory = sv.reg : !hw.inout<uarray<16xi8>>
  dbg.variable "Memory", %Memory mem : !hw.inout<uarray<16xi8>>
}
