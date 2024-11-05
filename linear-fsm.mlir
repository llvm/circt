fsm.machine @top(%mispec_x: i32) -> () attributes {initialState = "SpecSCC_NUM_fsm_Init0_ar"} {

	%rbwe = fsm.variable "rbwe" {initValue = 0 : i32} : i32
	%selSlowPath_x = fsm.variable "selSlowPath_x" {initValue = 0 : i32} : i32

	%0 = hw.constant 0 : i32
	%1 = hw.constant 1 : i32

	%4 = hw.constant 4 : i32
	%5 = hw.constant 5 : i32

	
    
	fsm.state @SpecSCC_NUM_fsm_Init0_st output  {
		} transitions {
		fsm.transition @SpecSCC_NUM_fsm_Init1_ar guard {
					} action {
		}
	}
	fsm.state @SpecSCC_NUM_fsm_Init1_st output  {
		} transitions {
		fsm.transition @SpecSCC_NUM_fsm_Init2_ar guard {
					} action {
		}
	}
	fsm.state @SpecSCC_NUM_fsm_Init2_st output  {
		} transitions {
		fsm.transition @SpecSCC_NUM_fsm_Proceed_ar guard {
					} action {
		}
	}
	fsm.state @SpecSCC_NUM_fsm_Proceed_ar output  {
		} transitions {
		fsm.transition @SpecSCC_NUM_fsm_Proceed_ar guard {
					} action {
			fsm.update %rbwe, %1 : i32
			fsm.update %selSlowPath_x, %0 : i32
		}
	}

	fsm.state @SpecSCC_NUM_fsm_Init0_ar output  {
		} transitions {
		fsm.transition @SpecSCC_NUM_fsm_Init0_st guard {
					} action {
			fsm.update %rbwe, %1 : i32
			fsm.update %selSlowPath_x, %0 : i32
		}
	}
	fsm.state @SpecSCC_NUM_fsm_Init1_ar output  {
		} transitions {
		fsm.transition @SpecSCC_NUM_fsm_Init1_st guard {
					} action {
			fsm.update %rbwe, %1 : i32
			fsm.update %selSlowPath_x, %0 : i32
		}
	}
	fsm.state @SpecSCC_NUM_fsm_Init2_ar output  {
		} transitions {
		fsm.transition @SpecSCC_NUM_fsm_Init2_st guard {
					} action {
			fsm.update %rbwe, %1 : i32
			fsm.update %selSlowPath_x, %0 : i32
		}
	}
}