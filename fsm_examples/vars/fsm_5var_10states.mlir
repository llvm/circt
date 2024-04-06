fsm.machine @fsm5() -> () attributes {initialState = "_0"} {
	%x0 = fsm.variable "x0" {initValue = 0 : i16} : i16
	%x1 = fsm.variable "x1" {initValue = 0 : i16} : i16
	%x2 = fsm.variable "x2" {initValue = 0 : i16} : i16
	%x3 = fsm.variable "x3" {initValue = 0 : i16} : i16
	%x4 = fsm.variable "x4" {initValue = 0 : i16} : i16
	%c0 = hw.constant 0 : i16
	%c1 = hw.constant 1 : i16
	%c2 = hw.constant 2 : i16
	%c3 = hw.constant 3 : i16
	%c4 = hw.constant 4 : i16
	%c5 = hw.constant 5 : i16
	%c6 = hw.constant 6 : i16
	%c7 = hw.constant 7 : i16
	%c8 = hw.constant 8 : i16
	%c9 = hw.constant 9 : i16


	fsm.state @_0 output {
	} transitions {
		fsm.transition @_1
		action {
				%tmp = comb.add %x5, %c8 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c8 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c8 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c8 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c8 : i16
				fsm.update x0, %tmp : i16
			}
	}

	fsm.state @_1 output {
	} transitions {
		fsm.transition @_2
		action {
				%tmp = comb.add %x5, %c5 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c5 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c5 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c5 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c5 : i16
				fsm.update x0, %tmp : i16
			}
	}

	fsm.state @_2 output {
	} transitions {
		fsm.transition @_3
		action {
				%tmp = comb.add %x5, %c1 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c1 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c1 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c1 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c1 : i16
				fsm.update x0, %tmp : i16
			}
	}

	fsm.state @_3 output {
	} transitions {
		fsm.transition @_4
		action {
				%tmp = comb.add %x5, %c2 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c2 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c2 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c2 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c2 : i16
				fsm.update x0, %tmp : i16
			}
	}

	fsm.state @_4 output {
	} transitions {
		fsm.transition @_5
		action {
				%tmp = comb.add %x5, %c3 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c3 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c3 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c3 : i16
				fsm.update x0, %tmp : i16
				%tmp = comb.add %x5, %c3 : i16
				fsm.update x0, %tmp : i16
			}
	}

	fsm.state @_5 output {
	} transitions {
	}
}