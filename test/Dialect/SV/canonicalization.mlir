// RUN: circt-opt -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s

// CHECK-LABEL: func @if_dead_condition(%arg0: i1) {
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.always posedge %arg0  {
// CHECK-NEXT:      sv.fwrite [[FD]], "Reachable1"
// CHECK-NEXT:      sv.fwrite [[FD]], "Reachable2"
// CHECK-NEXT:      sv.fwrite [[FD]], "Reachable3"
// CHECK-NEXT:      sv.fwrite [[FD]], "Reachable4"
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @if_dead_condition(%arg0: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.always posedge %arg0 {
    %true = hw.constant true
    %false = hw.constant false

    sv.if %true {}

    sv.if %false {
        sv.fwrite %fd, "Unreachable0"
    }

    sv.if %true {
      sv.fwrite %fd, "Reachable1"
    }

    sv.if %true {
      sv.fwrite %fd, "Reachable2"
    } else {
      sv.fwrite %fd, "Unreachable2"
    }

    sv.if %false {
      sv.fwrite %fd, "Unreachable3"
    } else {
      sv.fwrite %fd, "Reachable3"
    }

    sv.if %false {
      sv.fwrite %fd, "Unreachable4"
    } else {
      sv.fwrite %fd, "Reachable4"
    }
  }

  return
}

// CHECK-LABEL: func @empy_op(%arg0: i1) {
// CHECK-NOT:     sv.if
// CHECK-NOT:     sv.ifdef
// CHECK-NOT:     sv.ifdef.procedural
// CHECK-NOT:     sv.always
// CHECK-NOT:     sv.initial
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @empy_op(%arg0: i1) {
  sv.initial {
    sv.if %arg0 {}
    sv.if %arg0 {} else {}
    sv.ifdef.procedural "SYNTHESIS" {}
    sv.ifdef.procedural "SYNTHESIS" {} else {}
  }
  sv.ifdef "SYNTHESIS" {}
  sv.ifdef "SYNTHESIS" {} else {}
  sv.always posedge %arg0 {}
  sv.initial {}
  return
}

// CHECK-LABEL: func @invert_if(%arg0: i1) {
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.initial  {
// CHECK-NEXT:      %0 = comb.xor %arg0, %true : i1
// CHECK-NEXT:      sv.if %0  {
// CHECK-NEXT:        sv.fwrite [[FD]], "Foo"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @invert_if(%arg0: i1) {
  sv.initial {
    sv.if %arg0 {
    } else {
      %fd = hw.constant 0x80000002 : i32
      sv.fwrite %fd, "Foo"
    }
  }
  return
}

// CHECK-LABEL: hw.module @mux_to_cond_assign_f
// CHECK-NEXT:    %r = sv.reg  : !hw.inout<i2>
// CHECK-NEXT:    %0 = sv.read_inout %r : !hw.inout<i2>
// CHECK-NEXT:    sv.alwaysff(posedge %clock)  {
// CHECK-NEXT:      sv.if %c  {
// CHECK-NEXT:        sv.passign %r, %data : i2
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    hw.output %0
// CHECK-NEXT:  }
hw.module @mux_to_cond_assign_f(%clock: i1, %c: i1, %data: i2) -> (r: i2) {
  %r = sv.reg  : !hw.inout<i2>
  %1 = sv.read_inout %r : !hw.inout<i2>
  %0 = comb.mux %c, %data, %1 : i2
  sv.alwaysff(posedge %clock)  {
    sv.passign %r, %0 : i2
  }
  hw.output %1: i2
}

// CHECK-LABEL: hw.module @mux_to_cond_assign_t
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %r = sv.reg  : !hw.inout<i2>
// CHECK-NEXT:    %r3 = sv.reg  sym @r3 : !hw.inout<i2>
// CHECK-NEXT:    %0 = sv.read_inout %r : !hw.inout<i2>
// CHECK-NEXT:    sv.alwaysff(posedge %clock)  {
// CHECK-NEXT:      %1 = comb.xor %c, %true : i1
// CHECK-NEXT:      sv.if %1  {
// CHECK-NEXT:        sv.passign %r, %data : i2
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    hw.output %0
// CHECK-NEXT:  }
hw.module @mux_to_cond_assign_t(%clock: i1, %c: i1, %data: i2) -> (r: i2) {
  %r = sv.reg  : !hw.inout<i2>
  %r2 = sv.reg  : !hw.inout<i2>
  %r3 = sv.reg sym @r3 : !hw.inout<i2>
  %1 = sv.read_inout %r : !hw.inout<i2>
  %0 = comb.mux %c, %1, %data : i2
  sv.alwaysff(posedge %clock)  {
    sv.passign %r, %0 : i2
  }
  hw.output %1 : i2
}

// CHECK-LABEL; @immediate_assert_canonicalization
hw.module @assert_canonicalization(%clock: i1) {
  %true = hw.constant 1 : i1
  %false = hw.constant 0 : i1
  sv.always posedge %clock {
    // CHECK-NOT: sv.assert
    sv.assert %true, immediate message "assert"
    // CHECK-NOT: sv.assume
    sv.assume %true, immediate message "assume"
    // CHECK-NOT: sv.cover
    sv.cover %false, immediate
  }

  // CHECK-NOT: sv.assert.concurrent
  sv.assert.concurrent posedge %clock, %true
  // CHECK-NOT: sv.assume.concurrent
  sv.assume.concurrent posedge %clock, %true
  // CHECK-NOT: sv.cover.concurrent
  sv.cover.concurrent posedge %clock, %false
}

// CHECK-LABEL: @case_stmt
hw.module @case_stmt(%arg: i3) {
  %fd = hw.constant 0x80000002 : i32
  sv.initial {
    // CHECK: sv.case %arg
    sv.case casez %arg : i3
    case b000: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case %arg
    sv.case casez %arg : i3
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case casez %arg
    sv.case casez %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case casez %arg
    sv.case casez %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }


    // CHECK: sv.case %arg
    sv.case casex %arg : i3
    case b000: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case casex %arg
    sv.case casex %arg : i3
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case casez %arg
    sv.case casex %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case casex %arg
    sv.case casex %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }


    // CHECK: sv.case %arg
    sv.case %arg : i3
    case b000: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case %arg
    sv.case %arg : i3
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case %arg
    sv.case %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case %arg
    sv.case %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

  }

  }

// Remove read-only registers.
// CHECK-LABEL: hw.module @remove_reg
// CHECK-NEXT:  %r2 = sv.reg sym @r3 : !hw.inout<i2>
// CHECK-NEXT:  hw.output
hw.module @remove_reg(%input: i2){
  %r1 = sv.reg  : !hw.inout<i2>
  %r2 = sv.reg sym @r3 : !hw.inout<i2>
  sv.initial {
    sv.passign %r1, %input : i2
    sv.bpassign %r1, %input : i2
  }
}
