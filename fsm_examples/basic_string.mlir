// FSM recognizing language 1001, we want to check that it only recognizes those strings 

fsm.machine @top(%go: i1) -> (i1) attributes {initialState = "A"} {
    %c_0 = hw.constant 1 : i1

    fsm.state @A output  {
        %neg = comb.xor %go, %c_0 : i1
        fsm.output %neg : i1
    } transitions {
        fsm.transition @B guard {
            fsm.return %go
        }
    }

    fsm.state @B output  {
        %neg = comb.xor %go, %c_0 : i1
        fsm.output %neg : i1
    } transitions {
        fsm.transition @C guard {
            %neg = comb.xor %go, %c_0 : i1
            fsm.return %neg
        }
    }

    fsm.state @C output  {
        %neg = comb.xor %go, %c_0 : i1
        fsm.output %neg : i1
    } transitions {
        fsm.transition @D guard {
        %neg = comb.xor %go, %c_0 : i1
        fsm.return %neg
        }
    }

    fsm.state @D output  {
        fsm.output %go : i1
    } transitions {
        fsm.transition @D guard {
            fsm.return %go
        }
    }
}