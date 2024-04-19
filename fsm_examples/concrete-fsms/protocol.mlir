fsm.machine @top(%go: i16) -> () attributes {initialState = "I"} {
    %c_0 = hw.constant 0 : i16
    %c_1 = hw.constant 1 : i16
    %c_2 = hw.constant 2 : i16

    // idle - ARREADY
    fsm.state @I output  { 

    } transitions {
        fsm.transition @M guard {
            %eq = comb.icmp eq %go, %c_0 : i16
            fsm.return %eq
        } 
        fsm.transition @E guard {
            %eq = comb.icmp eq %go, %c_1 : i16
            fsm.return %eq
        } 
    }

    fsm.state @M output  { 
    } transitions {
        fsm.transition @M guard {
            %eq = comb.icmp eq %go, %c_2 : i16
            fsm.return %eq
        } 
        fsm.transition @E guard {
            %eq = comb.icmp eq %go, %c_2 : i16
            fsm.return %eq
        } 
    }

    fsm.state @E output  { 
    } transitions {
        fsm.transition @I guard {
            %eq = comb.icmp eq %go, %c_2 : i16
            fsm.return %eq
        }  
    }

}