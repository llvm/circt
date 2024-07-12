fsm.machine @fsm40() -> (i16) attributes {initialState = "_0"} {
	%x0 = fsm.variable "x0" {initValue = 0 : i16} : i16
	%c1 = hw.constant 1 : i16


	fsm.state @_0 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_1
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_1 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_2
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_2 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_3
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_3 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_4
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_4 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_5
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_5 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_6
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_6 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_7
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_7 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_8
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_8 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_9
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_9 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_10
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_10 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_11
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_11 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_12
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_12 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_13
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_13 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_14
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_14 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_15
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_15 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_16
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_16 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_17
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_17 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_18
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_18 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_19
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_19 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_20
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_20 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_21
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_21 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_22
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_22 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_23
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_23 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_24
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_24 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_25
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_25 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_26
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_26 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_27
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_27 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_28
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_28 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_29
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_29 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_30
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_30 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_31
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_31 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_32
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_32 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_33
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_33 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_34
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_34 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_35
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_35 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_36
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_36 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_37
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_37 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_38
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_38 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_39
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_39 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_40
		action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
	}

	fsm.state @_40 output {
		fsm.output %x0: i16
	} transitions {
	}
}