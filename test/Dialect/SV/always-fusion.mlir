// RUN: circt-opt -rtl-cleanup %s | FileCheck %s

//CHECK-LABEL: rtl.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
//CHECK-NEXT:   sv.fwrite "Middle\0A"
//CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:     sv.fwrite "A1"
//CHECK-NEXT:     sv.fwrite "A2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.alwaysff(posedge %arg1)  {
//CHECK-NEXT:     sv.fwrite "B1"
//CHECK-NEXT:     sv.fwrite "B2"
//CHECK-NEXT:   }
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

// CHECK-LABEL: rtl.module @ifdef_merge(%arg0: i1) {
// CHECK-NEXT:    sv.ifdef "FOO"  {
// CHECK-NEXT:      sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:        sv.fwrite "A1"
// CHECK-NEXT:        sv.fwrite "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
rtl.module @ifdef_merge(%arg0: i1) {
  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "A1"
    }
  }
  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "B1"
    }
  }
  rtl.output
}


// CHECK-LABEL: rtl.module @ifdef_proc_merge(%arg0: i1) {
// CHECK-NEXT:    sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:      %true = comb.constant(true) : i1
// CHECK-NEXT:      %0 = comb.xor %arg0, %true : i1
// CHECK-NEXT:      sv.ifdef.procedural "FOO"  {
// CHECK-NEXT:        sv.fwrite "A1"
// CHECK-NEXT:        sv.fwrite "%x"(%0) : i1
// CHECK-NEXT:      }
// CHECK-NEXT:    }
rtl.module @ifdef_proc_merge(%arg0: i1) {
  sv.alwaysff(posedge %arg0) {
    sv.ifdef.procedural "FOO" {
      sv.fwrite "A1"
    }
    %true = comb.constant(true) : i1
    %0 = comb.xor %arg0, %true : i1
    sv.ifdef.procedural "FOO" {
       sv.fwrite "%x"(%0) : i1
    }
  }
  rtl.output
}
