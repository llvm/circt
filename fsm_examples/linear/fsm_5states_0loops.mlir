fsm.machine @fsm5() -> (i1) attributes {initialState = "_0"} {
	%x0 = fsm.variable "x0" {initValue = 0 : i16} : i16
	%c1 = hw.constant 1 : i16
	%c5 = hw.constant 5 : i16


	fsm.state @_0 output {
		%tmp2 = comb.icmp eq %x0, %c5 : i16
		fsm.output %tmp2 : i1
	} transitions {
		fsm.transition @_1
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_1 output {
		%tmp2 = comb.icmp eq %x0, %c5 : i16
		fsm.output %tmp2 : i1
	} transitions {
		fsm.transition @_2
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_2 output {
		%tmp2 = comb.icmp eq %x0, %c5 : i16
		fsm.output %tmp2 : i1
	} transitions {
		fsm.transition @_3
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_3 output {
		%tmp2 = comb.icmp eq %x0, %c5 : i16
		fsm.output %tmp2 : i1
	} transitions {
		fsm.transition @_4
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_4 output {
		%tmp2 = comb.icmp eq %x0, %c5 : i16
		fsm.output %tmp2 : i1
		
	} transitions {
		fsm.transition @_5
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_5 output {
		%tmp2 = comb.icmp eq %x0, %c5 : i16
		fsm.output %tmp2 : i1
	} transitions {



	}
}