// https://github.com/lowRISC/opentitan/blob/master/hw/ip/aes/rtl/aes_ctr_fsm.sv
// Note: treating ctr_value as an input as it's defined by external comb logic (in an SV Assign)
// Property: !alert_o (2nd output) => (state == CTR_IDLE || state == CTR_INCR)

fsm.machine @fsm10(%incr_i: i1, %incr_err_i: i1, %mr_err_i: i1) -> (i1, i1, i1) attributes {initialState = "CTR_IDLE"} {
	%ctr_slice_idx = fsm.variable "ctr_slice_idx" {initValue = 0 : i16} : i16
    %ctr_carry = fsm.variable "ctr_carry" {initValue = 0 : i1} : i1
    %true = hw.constant true
    %false = hw.constant false
    %c0_i16 = hw.constant 0 : i16
    %c0_i1 = hw.constant 0 : i1
    %c-1_i16 = hw.constant -1 : i16
    %c1_i16 = hw.constant 1 : i16




    fsm.state @CTR_IDLE output  {
        fsm.output %true, %false, %false : i1, i1, i1
    } transitions  {
        fsm.transition @CTR_INCR guard {
            fsm.return %incr_i
        } action {
            fsm.update %ctr_slice_idx, %c0_i16 : i16
            fsm.update %ctr_carry, %true : i1
        }
        fsm.transition @CTR_ERROR guard {
            %or = comb.or %incr_err_i, %mr_err_i : i1
            fsm.return %or
        }
    }

    fsm.state @CTR_INCR output  {
        fsm.output %false, %false, %true : i1, i1, i1
    } transitions  {
        fsm.transition @CTR_IDLE guard {
            %eq = comb.icmp eq %c-1_i16, %ctr_slice_idx : i16
            fsm.return %eq
        } action {
            // Bodge to get MSB:
            fsm.update %ctr_carry, %c0_i1 : i1
            %c1_i16 = hw.constant 1 : i16
            %sum = comb.add %ctr_slice_idx, %c1_i16 : i16
            fsm.update %ctr_slice_idx, %sum : i16
        }
        fsm.transition @CTR_INCR guard {
            %nother_cond1 = comb.icmp ne %c-1_i16, %ctr_slice_idx : i16
            %or = comb.or %incr_err_i, %mr_err_i : i1
            %nother_cond2 = comb.xor %or, %true : i1
            %and_nconds = comb.and %nother_cond1, %nother_cond2 : i1
            fsm.return %and_nconds
        } action {
            // Bodge to get MSB:
            fsm.update %ctr_carry, %c0_i1 : i1
            %sum = comb.add %ctr_slice_idx, %c1_i16 : i16
            fsm.update %ctr_slice_idx, %sum : i16
        }
        fsm.transition @CTR_ERROR guard {
            %or = comb.or %incr_err_i, %mr_err_i : i1
            fsm.return %or
        }    
    }

    fsm.state @CTR_ERROR output  {
        fsm.output %false, %true, %false : i1, i1, i1
    }
}