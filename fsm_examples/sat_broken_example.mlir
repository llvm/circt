fsm.machine @top(%in: i16) -> (i16) attributes {initialState = "A"} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16
  %c_0 = hw.constant 0 : i16
  %c_1 = hw.constant 1 : i16
  %c_5 = hw.constant 5 : i16

  fsm.state @A output  {
    fsm.output %cnt : i16
  } transitions {
    fsm.transition @B 
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

  }

  fsm.state @C output  {
    fsm.output %cnt : i16
  } transitions {
    fsm.transition @D
  }

  fsm.state @D output  {
    fsm.output %cnt : i16
  } transitions {
  }

}