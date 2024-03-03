fsm.machine @fsm5() -> () {initialState = "0"} {
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
				%tmp = comb.icmp slt 7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul 9 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @9 guard {
				%tmp = comb.icmp ne x0, 0 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x0, 9 : i16
				fsm.update x3, %tmp : i16
			}
		fsm.transition @5 guard {
				%tmp = comb.icmp eq x3, 1 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x0, 3 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @5 guard {
				%tmp = comb.icmp ne x1, 1 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x2, 8 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @3 guard {
				%tmp = comb.icmp sle x3, 7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x1, 6 : i16
				fsm.update x0, %tmp : i16
			}
		fsm.transition @2 guard {
				%tmp = comb.icmp slt x2, 1 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x2, 2 : i16
				fsm.update x3, %tmp : i16
			}
		fsm.transition @5 guard {
				%tmp = comb.icmp ne x3, 8 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x3, 4 : i16
				fsm.update x1, %tmp : i16
			}
	}

	fsm.state @1 output {
	} transitions {
		fsm.transition @2 guard {
				%tmp = comb.icmp eq 0 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul 1 : i16
				fsm.update x3, %tmp : i16
			}
		fsm.transition @2 guard {
				%tmp = comb.icmp sle x3, 4 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x1, 5 : i16
				fsm.update x3, %tmp : i16
			}
		fsm.transition @4 guard {
				%tmp = comb.icmp slt x2, 6 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x2, 6 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @1 guard {
				%tmp = comb.icmp eq x0, 7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x1, 6 : i16
				fsm.update x1, %tmp : i16
			}
	}

	fsm.state @2 output {
	} transitions {
		fsm.transition @3 guard {
				%tmp = comb.icmp eq 6 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add 4 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @7 guard {
				%tmp = comb.icmp slt x3, 5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x0, 6 : i16
				fsm.update x1, %tmp : i16
			}
		fsm.transition @6 guard {
				%tmp = comb.icmp slt x1, 4 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x1, 7 : i16
				fsm.update x1, %tmp : i16
			}
	}

	fsm.state @3 output {
	} transitions {
		fsm.transition @4 guard {
				%tmp = comb.icmp slt 7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul 6 : i16
				fsm.update x1, %tmp : i16
			}
		fsm.transition @6 guard {
				%tmp = comb.icmp sle x1, 7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x0, 8 : i16
				fsm.update x0, %tmp : i16
			}
	}

	fsm.state @4 output {
	} transitions {
		fsm.transition @5 guard {
				%tmp = comb.icmp sle 1 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul 4 : i16
				fsm.update x0, %tmp : i16
			}
		fsm.transition @8 guard {
				%tmp = comb.icmp eq x0, 9 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x0, 7 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @6 guard {
				%tmp = comb.icmp slt x2, 9 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x3, 6 : i16
				fsm.update x3, %tmp : i16
			}
		fsm.transition @8 guard {
				%tmp = comb.icmp eq x3, 7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x0, 6 : i16
				fsm.update x3, %tmp : i16
			}
		fsm.transition @0 guard {
				%tmp = comb.icmp slt x0, 5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x2, 1 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @3 guard {
				%tmp = comb.icmp eq x3, 6 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x0, 6 : i16
				fsm.update x1, %tmp : i16
			}
		fsm.transition @6 guard {
				%tmp = comb.icmp sle x1, 8 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x0, 7 : i16
				fsm.update x0, %tmp : i16
			}
		fsm.transition @6 guard {
				%tmp = comb.icmp ne x2, 0 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x3, 1 : i16
				fsm.update x3, %tmp : i16
			}
	}

	fsm.state @5 output {
	} transitions {
		fsm.transition @6 guard {
				%tmp = comb.icmp slt 5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add 4 : i16
				fsm.update x1, %tmp : i16
			}
	}

	fsm.state @6 output {
	} transitions {
		fsm.transition @7 guard {
				%tmp = comb.icmp eq 5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add 8 : i16
				fsm.update x1, %tmp : i16
			}
		fsm.transition @5 guard {
				%tmp = comb.icmp eq x1, 5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x3, 1 : i16
				fsm.update x3, %tmp : i16
			}
		fsm.transition @4 guard {
				%tmp = comb.icmp sle x1, 4 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x0, 1 : i16
				fsm.update x0, %tmp : i16
			}
		fsm.transition @6 guard {
				%tmp = comb.icmp eq x3, 6 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x0, 0 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @1 guard {
				%tmp = comb.icmp slt x2, 7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x3, 9 : i16
				fsm.update x3, %tmp : i16
			}
		fsm.transition @1 guard {
				%tmp = comb.icmp sle x3, 1 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x2, 0 : i16
				fsm.update x3, %tmp : i16
			}
	}

	fsm.state @7 output {
	} transitions {
		fsm.transition @8 guard {
				%tmp = comb.icmp eq 4 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul 1 : i16
				fsm.update x0, %tmp : i16
			}
		fsm.transition @7 guard {
				%tmp = comb.icmp sle x2, 4 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x0, 8 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @2 guard {
				%tmp = comb.icmp ne x3, 3 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x1, 1 : i16
				fsm.update x0, %tmp : i16
			}
		fsm.transition @7 guard {
				%tmp = comb.icmp sle x1, 5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x1, 5 : i16
				fsm.update x0, %tmp : i16
			}
		fsm.transition @0 guard {
				%tmp = comb.icmp sle x2, 0 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x2, 0 : i16
				fsm.update x1, %tmp : i16
			}
		fsm.transition @3 guard {
				%tmp = comb.icmp slt x3, 7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x0, 2 : i16
				fsm.update x1, %tmp : i16
			}
		fsm.transition @8 guard {
				%tmp = comb.icmp eq x1, 8 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x1, 0 : i16
				fsm.update x0, %tmp : i16
			}
		fsm.transition @2 guard {
				%tmp = comb.icmp sle x2, 6 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x2, 6 : i16
				fsm.update x3, %tmp : i16
			}
	}

	fsm.state @8 output {
	} transitions {
		fsm.transition @9 guard {
				%tmp = comb.icmp eq 3 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul 6 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @2 guard {
				%tmp = comb.icmp eq x2, 8 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x2, 5 : i16
				fsm.update x1, %tmp : i16
			}
		fsm.transition @6 guard {
				%tmp = comb.icmp sle x3, 5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul x3, 2 : i16
				fsm.update x1, %tmp : i16
			}
	}

	fsm.state @9 output {
	} transitions {
		fsm.transition @6 guard {
				%tmp = comb.icmp slt x2, 4 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x2, 8 : i16
				fsm.update x2, %tmp : i16
			}
		fsm.transition @0 guard {
				%tmp = comb.icmp slt x0, 3 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add x3, 1 : i16
				fsm.update x3, %tmp : i16
			}
	}
}