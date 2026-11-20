// RUN: circt-translate %s --emit-uhdi | FileCheck %s

// The verifier does not parse `ext`, so hand-written IR can hold JSON that
// does not parse. That language is left out; the others stay.

// CHECK:      "profiles": [
// CHECK-NEXT:   "x-good"
// CHECK-NEXT: ],
// CHECK-NOT:  "x-bad"
// CHECK:      "name": "Top",
// CHECK-NEXT: "x-good": [
// CHECK-NEXT:   1
// CHECK-NEXT: ]
// CHECK-NOT:  "x-bad"
hw.module @Top(in %a : i8) attributes {dbg.moduleinfo = {name = "Top", ext = {bad = "{", good = "[1]"}}} {
  dbg.variable "a", %a : i8
}
