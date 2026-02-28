// RUN: circt-opt %s | FileCheck %s

hw.type_scope @enum_typedecls {
  hw.typedecl @enum0 : !hw.enum<A, B>
}

// CHECK-LABEL: hw.module @test_case_stmts() {
hw.module @test_case_stmts() {
  // CHECK: [[FD:%.*]] = hw.constant -2147483646 : i32
  %fd = hw.constant 0x80000002 : i32

  // Case bit pattern
  // CHECK-NEXT: [[COND:%.*]] = hw.constant true
  %0 = hw.constant 1 : i1
  sv.initial {
    sv.case %0 : i1
    case b0: {
      sv.fwrite %fd, "zero"
    }
    case b1: {
      sv.fwrite %fd, "one"
    }
    default: {
      sv.fwrite %fd, "default"
    }
  }

  // Case enum pattern
  %1 = hw.enum.constant A : !hw.typealias<@enum_typedecls::@enum0, !hw.enum<A, B>>
  sv.initial {
    sv.case %1 : !hw.typealias<@enum_typedecls::@enum0, !hw.enum<A, B>>
    case A: {
      sv.fwrite %fd, "A"
    }
    case B: {
      sv.fwrite %fd, "B"
    }
    default: {
      sv.fwrite %fd, "default"
    }
  }

  // Case expr pattern
  %foo = sv.wire : !hw.inout<i1>
  %bar = sv.wire : !hw.inout<i1>
  // CHECK: [[FOO:%.*]] = sv.read_inout %foo : !hw.inout<i1>
  // CHECK: [[BAR:%.*]] = sv.read_inout %bar : !hw.inout<i1>
  %2 = sv.read_inout %foo : !hw.inout<i1>
  %3 = sv.read_inout %bar : !hw.inout<i1>
  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT:   sv.case [[COND]] : i1
  // CHECK-NEXT:   case [[FOO]]: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "foo"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   case [[BAR]]: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "bar"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   default: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "default"
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sv.initial {
    sv.case %0 : i1
    case %2: {
      sv.fwrite %fd, "foo"
    }
    case %3: {
      sv.fwrite %fd, "bar"
    }
    default: {
      sv.fwrite %fd, "default"
    }
  }

  hw.output
}
