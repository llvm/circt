module {
  %c0_i16 = hw.constant 0 : i16
  %c1_i3 = hw.constant 1 : i3
  %c-1_i3 = hw.constant -1 : i3
  %c0_i3 = hw.constant 0 : i3
  %false = hw.constant 0 : i1
  %true = hw.constant 1 : i1
  %c0_i15 = hw.constant 0 : i15
  fsm.machine @aes_ctr_fsm(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i16) -> () attributes {initialState = "state_14"} {
    %ctr_slice_idx_q = fsm.variable "ctr_slice_idx_q" {initValue = 0 : i3} : i3
    %ctr_carry_q = fsm.variable "ctr_carry_q" {initValue = false} : i1
    fsm.state @state_14 output {
    } transitions {
      fsm.transition @state_14 guard {
        %0 = comb.or %arg1, %arg2 : i1
        %1 = comb.xor %arg0, %true : i1
        %2 = comb.xor %0, %true : i1
        %3 = comb.and %2, %1 : i1
        verif.assert %true : i1
        fsm.return %3
      } action {
        %0 = comb.mux %arg0, %c0_i3, %ctr_slice_idx_q : i3
        %1 = comb.or %arg0, %ctr_carry_q : i1
        verif.assert %true : i1
        fsm.update %ctr_carry_q, %1 : i1
        fsm.update %ctr_slice_idx_q, %0 : i3
      }
     
    }
    
  }
}
