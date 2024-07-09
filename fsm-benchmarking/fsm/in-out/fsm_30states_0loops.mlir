fsm.machine @fsm30(%err: i16) -> (i16) attributes {initialState = "_0"} {
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
	}

	fsm.state @ERR output {
		fsm.output %x0: i16
	} transitions {
	}
}