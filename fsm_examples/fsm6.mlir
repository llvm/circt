fsm.machine @fsm6() -> () {initialState = "0"} {
	%x0 = fsm.variable "x0" {initValue = 0 : i16} : i16
	%x1 = fsm.variable "x1" {initValue = 0 : i16} : i16
	%x2 = fsm.variable "x2" {initValue = 0 : i16} : i16
	%x3 = fsm.variable "x3" {initValue = 0 : i16} : i16
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


	fsm.state @0 output {
	} transitions {
		fsm.transition @1 guard {
				%tmp = comb.icmp eq 1 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add 2 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @1 guard {
				%tmp = comb.icmp slt x2, 7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x3, 9 : i16
				fsm.update x1, %tmp : i16
			}
	}

	fsm.state @1 output {
	} transitions {
		fsm.transition @2 guard {
				%tmp = comb.icmp ne 9 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add 8 : i16
				fsm.update x3, %tmp : i16
			}
		fsm.transition @2 guard {
				%tmp = comb.icmp slt x2, 2 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x2, 4 : i16
				fsm.update x1, %tmp : i16
			}
	}

	fsm.state @2 output {
	} transitions {
		fsm.transition @3 guard {
				%tmp = comb.icmp sle 2 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add 0 : i16
				fsm.update x3, %tmp : i16
			}
		fsm.transition @3 guard {
				%tmp = comb.icmp slt x0, 4 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x1, 9 : i16
				fsm.update x1, %tmp : i16
			}
	}

	fsm.state @3 output {
	} transitions {
		fsm.transition @4 guard {
				%tmp = comb.icmp slt 2 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add 6 : i16
				fsm.update x3, %tmp : i16
			}
		fsm.transition @4 guard {
				%tmp = comb.icmp slt x3, 7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x0, 6 : i16
				fsm.update x2, %tmp : i16
			}
	}

	fsm.state @4 output {
	} transitions {
		fsm.transition @5 guard {
				%tmp = comb.icmp eq 3 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add 7 : i16
				fsm.update x0, %tmp : i16
			}
		fsm.transition @5 guard {
				%tmp = comb.icmp slt x0, 7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x3, 7 : i16
				fsm.update x3, %tmp : i16
			}
	}

	fsm.state @5 output {
	} transitions {
		fsm.transition @6 guard {
				%tmp = comb.icmp ne 5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add 8 : i16
				fsm.update x1, %tmp : i16
			}
		fsm.transition @6 guard {
				%tmp = comb.icmp eq x3, 9 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x0, 3 : i16
				fsm.update x0, %tmp : i16
			}
	}

	fsm.state @6 output {
	} transitions {
		fsm.transition @7 guard {
				%tmp = comb.icmp ne 8 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul 2 : i16
				fsm.update x0, %tmp : i16
			}
	}

	fsm.state @7 output {
	} transitions {
		fsm.transition @8 guard {
				%tmp = comb.icmp eq 5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add 7 : i16
				fsm.update x1, %tmp : i16
			}
	}

	fsm.state @8 output {
	} transitions {
		fsm.transition @9 guard {
				%tmp = comb.icmp eq 9 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul 8 : i16
				fsm.update x0, %tmp : i16
			}
	}

	fsm.state @9 output {
	} transitions {
	}
}