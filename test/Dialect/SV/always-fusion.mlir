// RUN: circt-opt -sv-always-fusion %s | FileCheck %s

//CHECK-LABEL: rtl.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
//CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:     sv.fwrite "A1"
//CHECK-NEXT:     sv.fwrite "A2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.alwaysff(posedge %arg1)  {
//CHECK-NEXT:     sv.fwrite "B1"
//CHECK-NEXT:     sv.fwrite "B2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.fwrite "Middle\0A"
//CHECK-NEXT:   rtl.output
//CHECK-NEXT: }

rtl.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
  sv.alwaysff(posedge %arg0) {
    sv.fwrite "A1"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite "B1"
  }
  sv.fwrite "Middle\n"
  sv.alwaysff(posedge %arg0) {
    sv.fwrite "A2"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite "B2"
  }
  rtl.output
}

// CHECK-LABEL: rtl.module @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite "A1"
// CHECK-NEXT:     sv.fwrite "A2"
// CHECK-NEXT:   }(asyncreset : negedge %arg1)  {
// CHECK-NEXT:     sv.fwrite "B1"
// CHECK-NEXT:     sv.fwrite "B2"
// CHECK-NEXT:   }
// CHECK-NEXT:   rtl.output
// CHECK-NEXT: }

rtl.module @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A1"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A2"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B2"
  }
  rtl.output
}


// CHECK-LABEL: rtl.module @alwaysff_different_reset(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite "A1"
// CHECK-NEXT:     sv.fwrite "A2"
// CHECK-NEXT:   }(asyncreset : negedge %arg1)  {
// CHECK-NEXT:     sv.fwrite "B1"
// CHECK-NEXT:     sv.fwrite "B2"
// CHECK-NEXT:   }
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite "C1"
// CHECK-NEXT:     sv.fwrite "C2"
// CHECK-NEXT:   }(asyncreset : posedge %arg1)  {
// CHECK-NEXT:     sv.fwrite "D1"
// CHECK-NEXT:     sv.fwrite "D2"
// CHECK-NEXT:   }
// CHECK-NEXT:   rtl.output
// CHECK-NEXT: }

rtl.module @alwaysff_different_reset(%arg0: i1, %arg1: i1) {
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A1"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "C1"
  } ( asyncreset : posedge %arg1) {
    sv.fwrite "D1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A2"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B2"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "C2"
  } ( asyncreset : posedge %arg1) {
    sv.fwrite "D2"
  }
  rtl.output
}

//CHECK-LABEL: rtl.module @alwaysff_ifdef(%arg0: i1) {
//CHECK-NEXT:  sv.ifdef "FOO" {
//CHECK-NEXT:     sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:       sv.fwrite "A1"
//CHECK-NEXT:       sv.fwrite "B1"
//CHECK-NEXT:     }
//CHECK-NEXT:   }
//CHECK-NEXT:   rtl.output
//CHECK-NEXT: }

rtl.module @alwaysff_ifdef(%arg0: i1) {
  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "A1"
    }
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "B1"
    }
  }
  rtl.output
}
