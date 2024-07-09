fsm.machine @fsm170(%err: i16) -> (i16) attributes {initialState = "_0"} {
	%x0 = fsm.variable "x0" {initValue = 0 : i16} : i16
	%c1 = hw.constant 1 : i16
	%c0 = hw.constant 0 : i16


	fsm.state @_0 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_1
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_1 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_2
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_2 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_3
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_3 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_4
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_4 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_5
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_5 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_6
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_6 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_7
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_7 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_8
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_8 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_9
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_9 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_10
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_10 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_11
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_11 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_12
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_12 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_13
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_13 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_14
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_14 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_15
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_15 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_16
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_16 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_17
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_17 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_18
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_18 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_19
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_19 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_20
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_20 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_21
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_21 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_22
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_22 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_23
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_23 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_24
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_24 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_25
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_25 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_26
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_26 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_27
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_27 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_28
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_28 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_29
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_29 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_30
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_30 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_31
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_31 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_32
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_32 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_33
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_33 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_34
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_34 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_35
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_35 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_36
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_36 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_37
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_37 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_38
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_38 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_39
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_39 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_40
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_40 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_41
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_41 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_42
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_42 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_43
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_43 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_44
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_44 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_45
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_45 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_46
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_46 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_47
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_47 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_48
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_48 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_49
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_49 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_50
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_50 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_51
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_51 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_52
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_52 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_53
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_53 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_54
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_54 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_55
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_55 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_56
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_56 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_57
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_57 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_58
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_58 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_59
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_59 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_60
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_60 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_61
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_61 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_62
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_62 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_63
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_63 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_64
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_64 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_65
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_65 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_66
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_66 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_67
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_67 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_68
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_68 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_69
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_69 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_70
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_70 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_71
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_71 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_72
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_72 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_73
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_73 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_74
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_74 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_75
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_75 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_76
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_76 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_77
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_77 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_78
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_78 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_79
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_79 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_80
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_80 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_81
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_81 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_82
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_82 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_83
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_83 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_84
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_84 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_85
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_85 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_86
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_86 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_87
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_87 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_88
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_88 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_89
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_89 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_90
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_90 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_91
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_91 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_92
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_92 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_93
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_93 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_94
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_94 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_95
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_95 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_96
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_96 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_97
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_97 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_98
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_98 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_99
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_99 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_100
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_100 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_101
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_101 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_102
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_102 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_103
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_103 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_104
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_104 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_105
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_105 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_106
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_106 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_107
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_107 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_108
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_108 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_109
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_109 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_110
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_110 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_111
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_111 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_112
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_112 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_113
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_113 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_114
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_114 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_115
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_115 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_116
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_116 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_117
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_117 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_118
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_118 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_119
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_119 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_120
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_120 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_121
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_121 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_122
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_122 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_123
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_123 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_124
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_124 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_125
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_125 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_126
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_126 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_127
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_127 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_128
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_128 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_129
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_129 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_130
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_130 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_131
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_131 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_132
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_132 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_133
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_133 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_134
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_134 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_135
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_135 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_136
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_136 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_137
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_137 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_138
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_138 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_139
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_139 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_140
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_140 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_141
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_141 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_142
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_142 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_143
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_143 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_144
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_144 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_145
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_145 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_146
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_146 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_147
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_147 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_148
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_148 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_149
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_149 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_150
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_150 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_151
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_151 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_152
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_152 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_153
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_153 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_154
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_154 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_155
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_155 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_156
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_156 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_157
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_157 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_158
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_158 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_159
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_159 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_160
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_160 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_161
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_161 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_162
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_162 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_163
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_163 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_164
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_164 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_165
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_165 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_166
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_166 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_167
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_167 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_168
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_168 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_169
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_169 output {
		fsm.output %x0: i16
	} transitions {
		fsm.transition @_170
			guard {
				%tmp1 = comb.icmp ne %err, %c0 : i16
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1 : i16
				fsm.return %tmp1
			}
	}

	fsm.state @_170 output {
		fsm.output %x0: i16
	} transitions {
	}

	fsm.state @ERR output {
		fsm.output %x0: i16
	} transitions {
	}
}