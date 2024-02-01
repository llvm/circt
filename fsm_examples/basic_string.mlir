// FSM recognizing language 1001, we want to check that it only recognizes those strings 

fsm.machine @top(%go: i1) -> () attributes {initialState = "A"} {
    %c_0 = hw.constant 1 : i1

    fsm.state @A output  {
    } transitions {
    fsm.transition @B guard {
        fsm.return %go
    }
    fsm.transition @A guard {
        %neg = comb.xor %go, %c_0 : i1
        fsm.return %neg
    }
    }

    fsm.state @B output  {
    } transitions {
    fsm.transition @A guard {
        fsm.return %go
    }
    fsm.transition @C guard {
        %neg = comb.xor %go, %c_0 : i1
        fsm.return %neg
    }
    }

    fsm.state @C output  {
    } transitions {
        fsm.transition @A guard {
        fsm.return %go
        }
        fsm.transition @D guard {
        %neg = comb.xor %go, %c_0 : i1
        fsm.return %neg
        }
    }

    fsm.state @D output  {
    } transitions {
    fsm.transition @D guard {
        fsm.return %go
    }
    fsm.transition @A guard {
        %neg = comb.xor %go, %c_0 : i1
        fsm.return %neg
    }
    }

}