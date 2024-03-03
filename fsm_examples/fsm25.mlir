fsm.machine @fsm25() -> () attributes {initialState = "_0"} {
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


	fsm.state @_0 output {
	} transitions {
		fsm.transition @_1 guard {
				%tmp = comb.icmp eq %x2, %c3 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul %x1, %c3 : i16
				fsm.update %x2, %tmp : i16
			}
	}

	fsm.state @_1 output {
	} transitions {
		fsm.transition @_2 guard {
				%tmp = comb.icmp ne %x1, %c6 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul %x2, %c9 : i16
				fsm.update %x1, %tmp : i16
			}
	}

	fsm.state @_2 output {
	} transitions {
		fsm.transition @_3 guard {
				%tmp = comb.icmp eq %x1, %c4 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul %x3, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_3 output {
	} transitions {
		fsm.transition @_4 guard {
				%tmp = comb.icmp eq %x2, %c8 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x2, %c7 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_4 output {
	} transitions {
		fsm.transition @_5 guard {
				%tmp = comb.icmp eq %x3, %c1 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul %x0, %c4 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_5 output {
	} transitions {
		fsm.transition @_6 guard {
				%tmp = comb.icmp eq %x0, %c4 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x3, %c2 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_6 output {
	} transitions {
		fsm.transition @_7 guard {
				%tmp = comb.icmp eq %x1, %c0 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul %x2, %c4 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_7 output {
	} transitions {
		fsm.transition @_8 guard {
				%tmp = comb.icmp ne %x0, %c9 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x2, %c0 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_10 guard {
				%tmp = comb.icmp ne %x2, %c9 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul %x3, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_8 output {
	} transitions {
		fsm.transition @_9 guard {
				%tmp = comb.icmp ne %x3, %c5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul %x2, %c0 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_9 output {
	} transitions {
		fsm.transition @_10 guard {
				%tmp = comb.icmp eq %x1, %c0 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul %x3, %c3 : i16
				fsm.update %x3, %tmp : i16
			}
	}

	fsm.state @_10 output {
	} transitions {
		fsm.transition @_11 guard {
				%tmp = comb.icmp eq %x3, %c2 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x1, %c9 : i16
				fsm.update %x1, %tmp : i16
			}
	}

	fsm.state @_11 output {
	} transitions {
		fsm.transition @_12 guard {
				%tmp = comb.icmp eq %x0, %c3 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x2, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_12 output {
	} transitions {
		fsm.transition @_13 guard {
				%tmp = comb.icmp eq %x0, %c7 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x1, %c2 : i16
				fsm.update %x1, %tmp : i16
			}
	}

	fsm.state @_13 output {
	} transitions {
		fsm.transition @_14 guard {
				%tmp = comb.icmp ne %x1, %c4 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c4 : i16
				fsm.update %x2, %tmp : i16
			}
	}

	fsm.state @_14 output {
	} transitions {
		fsm.transition @_15 guard {
				%tmp = comb.icmp eq %x2, %c5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x3, %c6 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_15 output {
	} transitions {
		fsm.transition @_16 guard {
				%tmp = comb.icmp eq %x3, %c2 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x1, %c5 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_16 output {
	} transitions {
		fsm.transition @_17 guard {
				%tmp = comb.icmp ne %x3, %c8 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul %x0, %c1 : i16
				fsm.update %x1, %tmp : i16
			}
	}

	fsm.state @_17 output {
	} transitions {
		fsm.transition @_18 guard {
				%tmp = comb.icmp eq %x3, %c3 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x1, %c3 : i16
				fsm.update %x1, %tmp : i16
			}
	}

	fsm.state @_18 output {
	} transitions {
		fsm.transition @_19 guard {
				%tmp = comb.icmp eq %x2, %c4 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x2, %c8 : i16
				fsm.update %x3, %tmp : i16
			}
	}

	fsm.state @_19 output {
	} transitions {
		fsm.transition @_20 guard {
				%tmp = comb.icmp ne %x1, %c9 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul %x0, %c9 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_20 output {
	} transitions {
		fsm.transition @_21 guard {
				%tmp = comb.icmp eq %x1, %c2 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x3, %c1 : i16
				fsm.update %x2, %tmp : i16
			}
	}

	fsm.state @_21 output {
	} transitions {
		fsm.transition @_22 guard {
				%tmp = comb.icmp ne %x2, %c8 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x3, %c3 : i16
				fsm.update %x3, %tmp : i16
			}
	}

	fsm.state @_22 output {
	} transitions {
		fsm.transition @_23 guard {
				%tmp = comb.icmp ne %x2, %c1 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.mul %x1, %c5 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_23 output {
	} transitions {
		fsm.transition @_24 guard {
				%tmp = comb.icmp eq %x0, %c3 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x1, %c8 : i16
				fsm.update %x1, %tmp : i16
			}
	}

	fsm.state @_24 output {
	} transitions {
	}
}