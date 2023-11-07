fsm.machine @m1(%go: i1) -> (i16) attributes {initialState = "A"} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16
  %c_0 = hw.constant 0 : i16
  %c_1 = hw.constant 1 : i16
  %c_5 = hw.constant 5 : i16

  fsm.state @A output  {
    fsm.output %cnt : i16
  } transitions {
    fsm.transition @B guard {
      fsm.return %go
    }
  }

  fsm.state @B output  {
    fsm.output %cnt : i16
  } transitions {
    fsm.transition @A guard {
      %eq = comb.icmp eq %cnt, %c_5 : i16
      fsm.return %eq
    } action {
      fsm.update %cnt, %c_0 : i16
    }

    fsm.transition @B action {
      %add1 = comb.add %cnt, %c_1 : i16
      fsm.update %cnt, %add1 : i16
    }
  }
}

fsm.machine @m2(%cnt2: i16) -> (i1) attributes {initialState = "C"} {
  %ack = fsm.variable "ack" {initValue = 0 : i1} : i1
  %c_0 = hw.constant 0 : i16
  %c_1 = hw.constant 1 : i1

  fsm.state @C output  {
    fsm.output %ack : i1
  } transitions {
    fsm.transition @D guard {
      %eq = comb.icmp eq %cnt2, %c_0 : i16
      fsm.return %eq
    } action {
        %add1 = comb.add %ack, %c_1 : i1
        fsm.update %ack, %add1 : i1
    }
  }

  fsm.state @D output  {
    fsm.output %ack : i1
  } transitions {
    
  }
}

hw.module @counter(in %clk: !seq.clock, in %rst_m1: i1) {
  %in = hw.constant true
  %out = fsm.hw_instance "m1_inst" @m1(%in), clock %clk, reset %rst_m1 : (i1) -> i16
  %out2 =fsm.hw_instance "m2_inst" @m2(%out), clock %clk, reset %rst_m1 : (i16) -> i1
}



