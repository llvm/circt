fsm.machine @fsm40() -> () attributes {initialState = "_0"} {
	%c2 = hw.constant 2 : i16
	%c5 = hw.constant 5 : i16
	%c6 = hw.constant 6 : i16
	%c6 = hw.constant 6 : i16
	%c11 = hw.constant 11 : i16
	%c12 = hw.constant 12 : i16
	%c11 = hw.constant 11 : i16
	%c10 = hw.constant 10 : i16
	%c11 = hw.constant 11 : i16
	%c14 = hw.constant 14 : i16
	%c20 = hw.constant 20 : i16
	%c18 = hw.constant 18 : i16
	%c18 = hw.constant 18 : i16
	%c27 = hw.constant 27 : i16
	%c27 = hw.constant 27 : i16
	%c28 = hw.constant 28 : i16
	%c30 = hw.constant 30 : i16
	%c30 = hw.constant 30 : i16
	%c26 = hw.constant 26 : i16
	%c38 = hw.constant 38 : i16
	%c36 = hw.constant 36 : i16
	%c31 = hw.constant 31 : i16
	%c32 = hw.constant 32 : i16
	%c42 = hw.constant 42 : i16
	%c32 = hw.constant 32 : i16
	%c45 = hw.constant 45 : i16
	%c41 = hw.constant 41 : i16
	%c51 = hw.constant 51 : i16
	%c37 = hw.constant 37 : i16
	%c50 = hw.constant 50 : i16
	%c54 = hw.constant 54 : i16
	%c40 = hw.constant 40 : i16
	%c62 = hw.constant 62 : i16
	%c58 = hw.constant 58 : i16
	%c46 = hw.constant 46 : i16
	%c38 = hw.constant 38 : i16
	%c47 = hw.constant 47 : i16
	%c1 = hw.constant 1 : i16


	fsm.state @_0 output {
	} transitions {
		fsm.transition @_1
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_1 output {
	} transitions {
		fsm.transition @_2
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_2 output {
	} transitions {
		fsm.transition @_3
			guard {
				%tmp = comb.icmp uge %x0, %c2 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_0
			guard {
				%tmp = comb.icmp ult %x0, %c2 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_3 output {
	} transitions {
		fsm.transition @_4
			guard {
				%tmp = comb.icmp uge %x0, %c5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_1
			guard {
				%tmp = comb.icmp ult %x0, %c5 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_4 output {
	} transitions {
		fsm.transition @_5
			guard {
				%tmp = comb.icmp uge %x0, %c6 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_1
			guard {
				%tmp = comb.icmp ult %x0, %c6 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_5 output {
	} transitions {
		fsm.transition @_6
			guard {
				%tmp = comb.icmp uge %x0, %c6 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_1
			guard {
				%tmp = comb.icmp ult %x0, %c6 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_6 output {
	} transitions {
		fsm.transition @_7
			guard {
				%tmp = comb.icmp uge %x0, %c11 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_4
			guard {
				%tmp = comb.icmp ult %x0, %c11 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_7 output {
	} transitions {
		fsm.transition @_8
			guard {
				%tmp = comb.icmp uge %x0, %c12 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_0
			guard {
				%tmp = comb.icmp ult %x0, %c12 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_8 output {
	} transitions {
		fsm.transition @_9
			guard {
				%tmp = comb.icmp uge %x0, %c11 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_5
			guard {
				%tmp = comb.icmp ult %x0, %c11 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_9 output {
	} transitions {
		fsm.transition @_10
			guard {
				%tmp = comb.icmp uge %x0, %c10 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_0
			guard {
				%tmp = comb.icmp ult %x0, %c10 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_10 output {
	} transitions {
		fsm.transition @_11
			guard {
				%tmp = comb.icmp uge %x0, %c11 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_3
			guard {
				%tmp = comb.icmp ult %x0, %c11 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_11 output {
	} transitions {
		fsm.transition @_12
			guard {
				%tmp = comb.icmp uge %x0, %c14 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_7
			guard {
				%tmp = comb.icmp ult %x0, %c14 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_12 output {
	} transitions {
		fsm.transition @_13
			guard {
				%tmp = comb.icmp uge %x0, %c20 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_1
			guard {
				%tmp = comb.icmp ult %x0, %c20 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_13 output {
	} transitions {
		fsm.transition @_14
			guard {
				%tmp = comb.icmp uge %x0, %c18 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_8
			guard {
				%tmp = comb.icmp ult %x0, %c18 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_14 output {
	} transitions {
		fsm.transition @_15
			guard {
				%tmp = comb.icmp uge %x0, %c18 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_9
			guard {
				%tmp = comb.icmp ult %x0, %c18 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_15 output {
	} transitions {
		fsm.transition @_16
			guard {
				%tmp = comb.icmp uge %x0, %c27 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_2
			guard {
				%tmp = comb.icmp ult %x0, %c27 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_16 output {
	} transitions {
		fsm.transition @_17
			guard {
				%tmp = comb.icmp uge %x0, %c27 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_0
			guard {
				%tmp = comb.icmp ult %x0, %c27 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_17 output {
	} transitions {
		fsm.transition @_18
			guard {
				%tmp = comb.icmp uge %x0, %c28 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_12
			guard {
				%tmp = comb.icmp ult %x0, %c28 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_18 output {
	} transitions {
		fsm.transition @_19
			guard {
				%tmp = comb.icmp uge %x0, %c30 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_12
			guard {
				%tmp = comb.icmp ult %x0, %c30 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_19 output {
	} transitions {
		fsm.transition @_20
			guard {
				%tmp = comb.icmp uge %x0, %c30 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_12
			guard {
				%tmp = comb.icmp ult %x0, %c30 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_20 output {
	} transitions {
		fsm.transition @_21
			guard {
				%tmp = comb.icmp uge %x0, %c26 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_15
			guard {
				%tmp = comb.icmp ult %x0, %c26 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_21 output {
	} transitions {
		fsm.transition @_22
			guard {
				%tmp = comb.icmp uge %x0, %c38 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_15
			guard {
				%tmp = comb.icmp ult %x0, %c38 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_22 output {
	} transitions {
		fsm.transition @_23
			guard {
				%tmp = comb.icmp uge %x0, %c36 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_20
			guard {
				%tmp = comb.icmp ult %x0, %c36 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_23 output {
	} transitions {
		fsm.transition @_24
			guard {
				%tmp = comb.icmp uge %x0, %c31 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_19
			guard {
				%tmp = comb.icmp ult %x0, %c31 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_24 output {
	} transitions {
		fsm.transition @_25
			guard {
				%tmp = comb.icmp uge %x0, %c32 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_8
			guard {
				%tmp = comb.icmp ult %x0, %c32 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_25 output {
	} transitions {
		fsm.transition @_26
			guard {
				%tmp = comb.icmp uge %x0, %c42 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_13
			guard {
				%tmp = comb.icmp ult %x0, %c42 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_26 output {
	} transitions {
		fsm.transition @_27
			guard {
				%tmp = comb.icmp uge %x0, %c32 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_6
			guard {
				%tmp = comb.icmp ult %x0, %c32 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_27 output {
	} transitions {
		fsm.transition @_28
			guard {
				%tmp = comb.icmp uge %x0, %c45 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_23
			guard {
				%tmp = comb.icmp ult %x0, %c45 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_28 output {
	} transitions {
		fsm.transition @_29
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_29 output {
	} transitions {
		fsm.transition @_30
			guard {
				%tmp = comb.icmp uge %x0, %c41 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_18
			guard {
				%tmp = comb.icmp ult %x0, %c41 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_30 output {
	} transitions {
		fsm.transition @_31
			guard {
				%tmp = comb.icmp uge %x0, %c51 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_12
			guard {
				%tmp = comb.icmp ult %x0, %c51 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_31 output {
	} transitions {
		fsm.transition @_32
			guard {
				%tmp = comb.icmp uge %x0, %c37 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_2
			guard {
				%tmp = comb.icmp ult %x0, %c37 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_32 output {
	} transitions {
		fsm.transition @_33
			guard {
				%tmp = comb.icmp uge %x0, %c50 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_9
			guard {
				%tmp = comb.icmp ult %x0, %c50 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_33 output {
	} transitions {
		fsm.transition @_34
			guard {
				%tmp = comb.icmp uge %x0, %c54 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_21
			guard {
				%tmp = comb.icmp ult %x0, %c54 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_34 output {
	} transitions {
		fsm.transition @_35
			guard {
				%tmp = comb.icmp uge %x0, %c40 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_2
			guard {
				%tmp = comb.icmp ult %x0, %c40 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_35 output {
	} transitions {
		fsm.transition @_36
			guard {
				%tmp = comb.icmp uge %x0, %c62 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_3
			guard {
				%tmp = comb.icmp ult %x0, %c62 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_36 output {
	} transitions {
		fsm.transition @_37
			guard {
				%tmp = comb.icmp uge %x0, %c58 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_24
			guard {
				%tmp = comb.icmp ult %x0, %c58 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_37 output {
	} transitions {
		fsm.transition @_38
			guard {
				%tmp = comb.icmp uge %x0, %c46 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_11
			guard {
				%tmp = comb.icmp ult %x0, %c46 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_38 output {
	} transitions {
		fsm.transition @_39
			guard {
				%tmp = comb.icmp uge %x0, %c38 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_12
			guard {
				%tmp = comb.icmp ult %x0, %c38 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_39 output {
	} transitions {
		fsm.transition @_40
			guard {
				%tmp = comb.icmp uge %x0, %c47 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @_31
			guard {
				%tmp = comb.icmp ult %x0, %c47 : i16
				fsm.return %tmp
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_nr output {
	} transitions {
		fsm.transition @_0
	}
}