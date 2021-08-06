// RUN: circt-opt -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s

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
    %true = hw.constant true
    %false = hw.constant false

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

// CHECK-LABEL: func @empy_op(%arg0: i1) {
// CHECK-NOT:     sv.if
// CHECK-NOT:     sv.ifdef
// CHECK-NOT:     sv.always
// CHECK-NOT:     sv.initial
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func @empy_op(%arg0: i1) {
  sv.initial {
    sv.if %arg0 {}
    sv.if %arg0 {} else {}
  }
  sv.ifdef "SYNTHESIS" {}
  sv.ifdef "SYNTHESIS" {} else {}
  sv.always posedge %arg0 {}
  sv.initial {}
  return
}

// CHECK-LABEL: func @invert_if(%arg0: i1) {
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    sv.initial  {
// CHECK-NEXT:      %0 = comb.xor %arg0, %true : i1
// CHECK-NEXT:      sv.if %0  {
// CHECK-NEXT:        sv.fwrite "Foo"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func @invert_if(%arg0: i1) {
  sv.initial {
    sv.if %arg0 {
    } else {
      sv.fwrite "Foo"
    }
  }
  return
}

// CHECK-LABEL: func @mux_to_cond_assign_f
// CHECK-NEXT:    %r = sv.reg  : !hw.inout<i2>
// CHECK-NEXT:    sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:      sv.if %arg1  {
// CHECK-NEXT:        sv.passign %r, %arg2 : i2
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func @mux_to_cond_assign_f(%clock: i1, %c: i1, %data: i2) {
  %r = sv.reg  : !hw.inout<i2>
  %1 = sv.read_inout %r : !hw.inout<i2>
  %0 = comb.mux %c, %data, %1 : i2
  sv.alwaysff(posedge %clock)  {
    sv.passign %r, %0 : i2
  }
  return
}

// CHECK-LABEL: func @mux_to_cond_assign_t
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %r = sv.reg  : !hw.inout<i2>
// CHECK-NEXT:    %r3 = sv.reg  sym @r3 : !hw.inout<i2>
// CHECK-NEXT:    sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:      %0 = comb.xor %arg1, %true : i1
// CHECK-NEXT:      sv.if %0  {
// CHECK-NEXT:        sv.passign %r, %arg2 : i2
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func @mux_to_cond_assign_t(%clock: i1, %c: i1, %data: i2) {
  %r = sv.reg  : !hw.inout<i2>
  %r2 = sv.reg  : !hw.inout<i2>
  %r3 = sv.reg sym @r3 : !hw.inout<i2>
  %1 = sv.read_inout %r : !hw.inout<i2>
  %0 = comb.mux %c, %1, %data : i2
  sv.alwaysff(posedge %clock)  {
    sv.passign %r, %0 : i2
  }
  return
}
