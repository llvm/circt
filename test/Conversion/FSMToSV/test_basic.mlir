// RUN: circt-opt -split-input-file -convert-fsm-to-sv %s | FileCheck %s

fsm.machine @FSM(%arg0: i1, %arg1: i1) -> (i8) attributes {initialState = "A"} {
  %c_0 = hw.constant 0 : i8
  fsm.state @A output  {
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @B
  }

  fsm.state @B output  {
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @A
  }
}

// CHECK: hw.module @top(%arg0: i1, %arg1: i1, %clk: i1, %rst: i1) -> (out: i8) {
// CHECK:   %fsm_inst.out0 = hw.instance "fsm_inst" @FSM(in0: %arg0: i1, in1: %arg1: i1, clk: %clk: i1, rst: %rst: i1) -> (out0: i8)
// CHECK:   hw.output %fsm_inst.out0 : i8
// CHECK: }
hw.module @top(%arg0: i1, %arg1: i1, %clk : i1, %rst : i1) -> (out: i8) {
    %out = fsm.hw_instance "fsm_inst" @FSM(%arg0, %arg1), clock %clk, reset %rst : (i1, i1) -> (i8)
    hw.output %out : i8
}

// -----

// CHECK:        hw.type_scope @fsm_enum_typedecls {
// CHECK-NEXT:     hw.typedecl @top_state_t : !hw.enum<A, B>
// CHECK-NEXT:   }

// CHECK-LABEL:  hw.module @top(%a0: i1, %a1: i1, %clk: i1, %rst: i1) -> (r0: i8, r1: i8) {
// CHECK-NEXT:    %A = hw.enum.constant A : !hw.typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>
// CHECK-NEXT:    %to_A = sv.reg sym @A  : !hw.inout<typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    sv.assign %to_A, %A : !hw.typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>
// CHECK-NEXT:    %0 = sv.read_inout %to_A : !hw.inout<typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    %B = hw.enum.constant B : !hw.typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>
// CHECK-NEXT:    %to_B = sv.reg sym @B  : !hw.inout<typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    sv.assign %to_B, %B : !hw.typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>
// CHECK-NEXT:    %1 = sv.read_inout %to_B : !hw.inout<typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    %state_next = sv.reg  : !hw.inout<typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    %2 = sv.read_inout %state_next : !hw.inout<typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>>
// CHECK-NEXT:    %state_reg = seq.compreg %2, %clk, %rst, %0  : !hw.typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>
// CHECK-NEXT:    %c42_i8 = hw.constant 42 : i8
// CHECK-NEXT:    %c0_i8 = hw.constant 0 : i8
// CHECK-NEXT:    %c1_i8 = hw.constant 1 : i8
// CHECK-NEXT:    %3 = comb.and %a0, %a1 : i1
// CHECK-NEXT:    %4 = comb.mux %3, %0, %1 : !hw.typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>
// CHECK-NEXT:    %output_0 = sv.reg  : !hw.inout<i8>
// CHECK-NEXT:    %output_1 = sv.reg  : !hw.inout<i8>
// CHECK-NEXT:    sv.alwayscomb {
// CHECK-NEXT:      sv.case %state_reg : !hw.typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>
// CHECK-NEXT:      case A: {
// CHECK-NEXT:        sv.bpassign %state_next, %1 : !hw.typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>
// CHECK-NEXT:        sv.bpassign %output_0, %c0_i8 : i8
// CHECK-NEXT:        sv.bpassign %output_1, %c42_i8 : i8
// CHECK-NEXT:      }
// CHECK-NEXT:      case B: {
// CHECK-NEXT:        sv.bpassign %state_next, %4 : !hw.typealias<@fsm_enum_typedecls::@top_state_t, !hw.enum<A, B>>
// CHECK-NEXT:        sv.bpassign %output_0, %c1_i8 : i8
// CHECK-NEXT:        sv.bpassign %output_1, %c42_i8 : i8
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    %5 = sv.read_inout %output_0 : !hw.inout<i8>
// CHECK-NEXT:    %6 = sv.read_inout %output_1 : !hw.inout<i8>
// CHECK-NEXT:    hw.output %5, %6 : i8, i8
// CHECK-NEXT:  }


fsm.machine @top(%a0: i1, %arg1: i1) -> (i8, i8) attributes {initialState = "A", argNames = ["a0", "a1"], resNames = ["r0", "r1"]} {
  %c_42 = hw.constant 42 : i8
  fsm.state @A output  {
    %c_0 = hw.constant 0 : i8
    fsm.output %c_0, %c_42 : i8, i8
  } transitions {
    fsm.transition @B
  }

  fsm.state @B output  {
    %c_1 = hw.constant 1 : i8
    fsm.output %c_1, %c_42 : i8, i8
  } transitions {
    fsm.transition @A guard {
      %g = comb.and %a0, %arg1 : i1
      fsm.return %g
    }
  }
}

// -----

// CHECK:       %[[CNT_ADD_1:.*]] = comb.add %cnt_reg, %c1_i16 : i16
// CHECK:       sv.alwayscomb {
// CHECK-NEXT:    sv.bpassign %cnt_next, %cnt_reg : i16
// CHECK-NEXT:    sv.case %state_reg : !hw.typealias<@fsm_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>
// CHECK-NEXT:    case A: {
// CHECK-NEXT:      sv.bpassign %state_next, %[[B:.*]] : !hw.typealias<@fsm_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>
// CHECK-NEXT:      sv.bpassign %output_0, %cnt_reg : i16
// CHECK-NEXT:    }
// CHECK-NEXT:    case B: {
// CHECK-NEXT:      sv.bpassign %state_next, %[[A:.*]] : !hw.typealias<@fsm_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>
// CHECK-NEXT:      sv.case %[[STATE_NEXT:.*]] : !hw.typealias<@fsm_enum_typedecls::@FSM_state_t, !hw.enum<A, B>>
// CHECK-NEXT:      case A: {
// CHECK-NEXT:        sv.bpassign %cnt_next, %[[CNT_ADD_1]] : i16
// CHECK-NEXT:      }
// CHECK-NEXT:      case B: {
// CHECK-NEXT:      }
// CHECK-NEXT:      sv.bpassign %output_0, %cnt_reg : i16
// CHECK-NEXT:    }
// CHECK-NEXT:  }

fsm.machine @FSM(%arg0: i1, %arg1: i1) -> (i16) attributes {initialState = "A"} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16
  %c_0 = hw.constant 0 : i16
  %c_1 = hw.constant 1 : i16
  fsm.state @A output  {
    fsm.output %cnt : i16
  } transitions {
    fsm.transition @B
  }

  fsm.state @B output  {
    fsm.output %cnt : i16
  } transitions {
    fsm.transition @A guard {
      fsm.return %arg0
    } action {
      %add1 = comb.add %cnt, %c_1 : i16
      fsm.update %cnt, %add1 : i16
    }
  }
}

// -----

// CHECK:      hw.type_scope @fsm_enum_typedecls {
// CHECK-NEXT:   hw.typedecl @M2_state_t : !hw.enum<A, B>
// CHECK-NEXT:   hw.typedecl @M1_state_t : !hw.enum<A, B>
// CHECK-NEXT: } {output_file = #hw.output_file<"fsm_enum_typedefs.sv">}

module {
  fsm.machine @M1() attributes {initialState = "A"} {
    fsm.state @A
    fsm.state @B
  }

  fsm.machine @M2() attributes {initialState = "A"} {
    fsm.state @A
    fsm.state @B
  }
}

// -----

// Test use of constants defined outside the machine. This is just a pass/fail
// test - the verifier will complain if the constant is not defined within
// the resulting hw.module.

module {
  %c0 = hw.constant 0 : i16
  fsm.machine @M1() -> (i16) attributes {initialState = "A"} {
    fsm.state @A output {
      fsm.output %c0 : i16
    }
    fsm.state @B output {
      fsm.output %c0 : i16
    }
  }

  fsm.machine @M2() -> (i16) attributes {initialState = "A"} {
    fsm.state @A output  {
      fsm.output %c0 : i16
    }
    fsm.state @B output {
      fsm.output %c0 : i16
    }
  }
}

// -----

// Test a machine which updates a variable based on inputs.
// Note: this machine replicates an FSM which activates different outputs
// in parallel.

// CHECK-LABEL:   hw.module @ParFSM

module {
  fsm.machine @ParFSM(%go : i1, %a_done : i1, %b_done : i1) -> (i1, i1 , i1) attributes {initialState = "idle"} {
    %c0 = hw.constant 0 : i1
    %c1 = hw.constant 1 : i1
    %c0_i2 = hw.constant 0 : i2
    %done = fsm.variable "done" {initValue = 0 : i2} : i2
    %a = comb.extract %done from 0 : (i2) -> i1
    %b = comb.extract %done from 1 : (i2) -> i1

    fsm.state @idle output {
      fsm.output %c0, %c0, %c1 : i1, i1, i1
    } transitions {
      fsm.transition @work guard {
        fsm.return %go
      }
    }

    fsm.state @work output {
      // Each output is active when the corresponding input has not yet fired.
      %a_out = comb.xor %a, %a_done : i1
      %b_out = comb.xor %b, %b_done : i1
      fsm.output %a_out, %b_out, %c0 : i1, i1, i1
    } transitions {
      // Go to idle when all inputs have fired.
      fsm.transition @idle guard {
        %all_done = comb.and %a, %b : i1
        fsm.return %all_done
      } action {
        // Reset the done variable.
        fsm.update %done, %c0_i2 : i2
      }

      // Else, always transition back to work, updating the done variables based
      // on the inputs.
      fsm.transition @work action {
        %a_next = comb.or %a, %a_done : i1
        %b_next = comb.or %b, %b_done : i1
        %done_next = comb.concat %a_next, %b_next : i1, i1
        fsm.update %done, %done_next : i2
      }
    }
  }
}
