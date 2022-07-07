// RUN: circt-opt -convert-fsm-to-sv %s | FileCheck %s
// CHECK:       hw.module.extern @single_result() -> (res: i8)

// CHECK:       hw.type_scope @FSM_enum_typedecls {
// CHECK-NEXT:    hw.typedecl @FSM_state_t : !hw.enum<A, B>
// CHECK-NEXT:  }

// CHECK:       hw.module @FSM(%in0: i1, %in1: i1, %clk: i1, %rst: i1) -> (out0: i8) {
// CHECK-NEXT:    %A = hw.enum.constant #hw.enum.field<A, !hw.typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    %to_A = sv.wire sym @A  : !hw.inout<typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    sv.assign %to_A, %A : !hw.typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>
// CHECK-NEXT:    %0 = sv.read_inout %to_A : !hw.inout<typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    %B = hw.enum.constant #hw.enum.field<B, !hw.typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    %to_B = sv.wire sym @B  : !hw.inout<typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    sv.assign %to_B, %B : !hw.typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>
// CHECK-NEXT:    %1 = sv.read_inout %to_B : !hw.inout<typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    %state_reg = seq.compreg %3, %clk, %rst, %0  : !hw.typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>
// CHECK-NEXT:    %some_wire = sv.wire  : !hw.inout<i8>
// CHECK-NEXT:    %2 = sv.read_inout %some_wire : !hw.inout<i8>
// CHECK-NEXT:    %b1.res = hw.instance "b1" @single_result() -> (res: i8)
// CHECK-NEXT:    sv.assign %some_wire, %b1.res : i8
// CHECK-NEXT:    %c0_i8 = hw.constant 0 : i8
// CHECK-NEXT:    sv.alwayscomb {
// CHECK-NEXT:      sv.case %state_reg : !hw.typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>
// CHECK-NEXT:      case A: {
// CHECK-NEXT:        sv.bpassign %next_state, %1 : !hw.typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>
// CHECK-NEXT:      }
// CHECK-NEXT:      case B: {
// CHECK-NEXT:        sv.bpassign %next_state, %0 : !hw.typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    %next_state = sv.reg  : !hw.inout<typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    %3 = sv.read_inout %next_state : !hw.inout<typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    sv.alwayscomb {
// CHECK-NEXT:      sv.case %state_reg : !hw.typealias<@FSM_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>
// CHECK-NEXT:      case A: {
// CHECK-NEXT:        sv.bpassign %output_0, %c0_i8 : i8
// CHECK-NEXT:      }
// CHECK-NEXT:      case B: {
// CHECK-NEXT:        sv.bpassign %output_0, %b1.res : i8
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    %output_0 = sv.reg  : !hw.inout<i8>
// CHECK-NEXT:    %4 = sv.read_inout %output_0 : !hw.inout<i8>
// CHECK-NEXT:    hw.output %4 : i8
// CHECK-NEXT:  }

// CHECK:       hw.module @top(%arg0: i1, %arg1: i1, %clk: i1, %rst: i1) -> (out: i8) {
// CHECK-NEXT:    %fsm_inst.out0 = hw.instance "fsm_inst" @FSM(in0: %arg0: i1, in1: %arg1: i1, clk: %clk: i1, rst: %rst: i1) -> (out0: i8)
// CHECK-NEXT:    hw.output %fsm_inst.out0 : i8
// CHECK-NEXT:  }


hw.module.extern @single_result() -> (res: i8)

fsm.machine @FSM(%arg0: i1, %arg1: i1) -> (i8) attributes {initialState = "A"} {
  %some_wire = sv.wire : !hw.inout<i8>
  %read = sv.read_inout %some_wire : !hw.inout<i8>
  %out1 = hw.instance "b1" @single_result() -> (res: i8)
  sv.assign %some_wire, %out1 : i8

  %c_0 = hw.constant 0 : i8

  fsm.state "A" output  {
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @B
  }

  fsm.state "B" output  {
    fsm.output %out1 : i8
  } transitions {
    fsm.transition @A
  }
}

hw.module @top(%arg0: i1, %arg1: i1, %clk : i1, %rst : i1) -> (out: i8) {
    %out = fsm.hw_instance "fsm_inst" @FSM(%arg0, %arg1), clock %clk, reset %rst : (i1, i1) -> (i8)
    hw.output %out : i8
}
