module {
  sv.verbatim "`include \22fsm_enum_typedefs.sv\22"
  hw.type_scope @fsm_enum_typedecls {
    hw.typedecl @m2_state_t : !hw.enum<C, D>
    hw.typedecl @m1_state_t : !hw.enum<A, B>
  } {output_file = #hw.output_file<"fsm_enum_typedefs.sv">}
  hw.module @m1(in %in0 : i1, out out0 : i16, in %clk : !seq.clock, in %rst : i1) {
    %A = hw.enum.constant A : !hw.typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>
    %to_A = sv.reg sym @A : !hw.inout<typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>> 
    sv.assign %to_A, %A : !hw.typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>
    %0 = sv.read_inout %to_A : !hw.inout<typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>>
    %B = hw.enum.constant B : !hw.typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>
    %to_B = sv.reg sym @B : !hw.inout<typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>> 
    sv.assign %to_B, %B : !hw.typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>
    %1 = sv.read_inout %to_B : !hw.inout<typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>>
    %state_next = sv.reg : !hw.inout<typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>> 
    %2 = sv.read_inout %state_next : !hw.inout<typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>>
    %state_reg = seq.compreg sym @state_reg  %2, %clk reset %rst, %0 : !hw.typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>  
    %cnt_next = sv.reg : !hw.inout<i16> 
    %c0_i16 = hw.constant 0 : i16
    %3 = sv.read_inout %cnt_next : !hw.inout<i16>
    %cnt_reg = seq.compreg sym @cnt_reg  %3, %clk reset %rst, %c0_i16 : i16  
    %c0_i16_0 = hw.constant 0 : i16
    %c1_i16 = hw.constant 1 : i16
    %c5_i16 = hw.constant 5 : i16
    %4 = comb.mux %in0, %1, %0 : !hw.typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>
    %5 = comb.icmp eq %cnt_reg, %c5_i16 : i16
    %6 = comb.add %cnt_reg, %c1_i16 : i16
    %7 = comb.mux %5, %0, %1 : !hw.typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>
    %output_0 = sv.reg : !hw.inout<i16> 
    sv.alwayscomb {
      sv.bpassign %cnt_next, %cnt_reg : i16
      sv.case %state_reg : !hw.typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>
      case A: {
        sv.bpassign %state_next, %4 : !hw.typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>
        sv.bpassign %output_0, %cnt_reg : i16
      }
      case B: {
        sv.bpassign %state_next, %7 : !hw.typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>
        sv.case %2 : !hw.typealias<@fsm_enum_typedecls::@m1_state_t, !hw.enum<A, B>>
        case A: {
          sv.bpassign %cnt_next, %c0_i16_0 : i16
        }
        case B: {
          sv.bpassign %cnt_next, %6 : i16
        }
        default: {
        }
        sv.bpassign %output_0, %cnt_reg : i16
      }
      default: {
      }
    }
    %8 = sv.read_inout %output_0 : !hw.inout<i16>
    hw.output %8 : i16
  }
  hw.module @m2(in %in0 : i16, out out0 : i1, in %clk : !seq.clock, in %rst : i1) {
    %C = hw.enum.constant C : !hw.typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>
    %to_C = sv.reg sym @C : !hw.inout<typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>> 
    sv.assign %to_C, %C : !hw.typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>
    %0 = sv.read_inout %to_C : !hw.inout<typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>>
    %D = hw.enum.constant D : !hw.typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>
    %to_D = sv.reg sym @D : !hw.inout<typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>> 
    sv.assign %to_D, %D : !hw.typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>
    %1 = sv.read_inout %to_D : !hw.inout<typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>>
    %state_next = sv.reg : !hw.inout<typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>> 
    %2 = sv.read_inout %state_next : !hw.inout<typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>>
    %state_reg = seq.compreg sym @state_reg  %2, %clk reset %rst, %0 : !hw.typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>  
    %ack_next = sv.reg : !hw.inout<i1> 
    %false = hw.constant false
    %3 = sv.read_inout %ack_next : !hw.inout<i1>
    %ack_reg = seq.compreg sym @ack_reg  %3, %clk reset %rst, %false : i1  
    %c0_i16 = hw.constant 0 : i16
    %true = hw.constant true
    %4 = comb.add %ack_reg, %true : i1
    %5 = comb.icmp eq %in0, %c0_i16 : i16
    %6 = comb.mux %5, %1, %0 : !hw.typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>
    %output_0 = sv.reg : !hw.inout<i1> 
    sv.alwayscomb {
      sv.bpassign %ack_next, %ack_reg : i1
      sv.case %state_reg : !hw.typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>
      case C: {
        sv.bpassign %state_next, %6 : !hw.typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>
        sv.case %2 : !hw.typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>
        case C: {
        }
        case D: {
          sv.bpassign %ack_next, %4 : i1
        }
        default: {
        }
        sv.bpassign %output_0, %ack_reg : i1
      }
      case D: {
        sv.bpassign %state_next, %1 : !hw.typealias<@fsm_enum_typedecls::@m2_state_t, !hw.enum<C, D>>
        sv.bpassign %output_0, %ack_reg : i1
      }
      default: {
      }
    }
    %7 = sv.read_inout %output_0 : !hw.inout<i1>
    hw.output %7 : i1
  }
  hw.module @counter(in %clk : !seq.clock, in %rst_m1 : i1) {
    %true = hw.constant true
    %m1_inst.out0 = hw.instance "m1_inst" @m1(in0: %true: i1, clk: %clk: !seq.clock, rst: %rst_m1: i1) -> (out0: i16)
    %m2_inst.out0 = hw.instance "m2_inst" @m2(in0: %m1_inst.out0: i16, clk: %clk: !seq.clock, rst: %rst_m1: i1) -> (out0: i1)
    hw.output
  }
}

