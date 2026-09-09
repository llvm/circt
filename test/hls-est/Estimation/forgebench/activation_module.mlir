module {
  memref.global @FM_buffer_4 : memref<64x28x28xf16> = uninitialized
  memref.global @FM_buffer_3 : memref<64x28x28xf16> = uninitialized
  memref.global @FM_buffer_2 : memref<64x28x28xf16> = uninitialized
  memref.global @FM_buffer_1 : memref<64x28x28xf16> = uninitialized
  func.func @_Z16load_feature_mapPA28_A28_fS1_(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          affine.store %0, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
  func.func @_Z17store_feature_mapPA28_A28_fS1_(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          affine.store %0, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
  func.func @_Z11compute_expPA28_A28_fS1_(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %1 = math.exp %0 : f16
          affine.store %1, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
  func.func @_Z8negativePA28_A28_fS1_(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %1 = arith.negf %0 : f16
          affine.store %1, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
  func.func @_Z11compute_divPA28_A28_fS1_S1_(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>, %arg2: memref<64x28x28xf16>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %0 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %1 = affine.load %arg1[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %2 = arith.divf %0, %1 : f16
          affine.store %2, %arg2[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
  func.func @_Z11compute_addPA28_A28_fS1_S1_(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>, %arg2: memref<64x28x28xf16>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %0 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %1 = affine.load %arg1[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %2 = arith.addf %0, %1 : f16
          affine.store %2, %arg2[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
  func.func @_Z11compute_mulPA28_A28_fS1_S1_(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>, %arg2: memref<64x28x28xf16>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %0 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %1 = affine.load %arg1[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %2 = arith.mulf %0, %1 : f16
          affine.store %2, %arg2[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
  func.func @_Z9set_valuePA28_A28_ff(%arg0: memref<64x28x28xf16>, %arg1: f16) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          affine.store %arg1, %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
  func.func @_Z6selectPA28_A28_fS1_S1_f(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>, %arg2: memref<64x28x28xf16>, %arg3: f16) {
    affine.for %arg4 = 0 to 64 {
      affine.for %arg5 = 0 to 28 {
        affine.for %arg6 = 0 to 28 {
          %0 = affine.load %arg0[%arg4, %arg5, %arg6] : memref<64x28x28xf16>
          %1 = arith.cmpf oge, %0, %arg3 : f16
          scf.if %1 {
            %2 = affine.load %arg0[%arg4, %arg5, %arg6] : memref<64x28x28xf16>
            affine.store %2, %arg2[%arg4, %arg5, %arg6] : memref<64x28x28xf16>
          } else {
            %2 = affine.load %arg1[%arg4, %arg5, %arg6] : memref<64x28x28xf16>
            affine.store %2, %arg2[%arg4, %arg5, %arg6] : memref<64x28x28xf16>
          }
        }
      }
    }
    return
  }
  func.func @_Z7sigmoidPA28_A28_fS1_(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>) {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = memref.get_global @FM_buffer_1 : memref<64x28x28xf16>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          affine.store %3, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    %1 = memref.get_global @FM_buffer_2 : memref<64x28x28xf16>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %4 = arith.negf %3 : f16
          affine.store %4, %1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %4 = math.exp %3 : f16
          affine.store %4, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    %2 = memref.get_global @FM_buffer_3 : memref<64x28x28xf16>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          affine.store %cst, %2[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %2[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %4 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %5 = arith.addf %3, %4 : f16
          affine.store %5, %1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %2[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %4 = affine.load %1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %5 = arith.divf %3, %4 : f16
          affine.store %5, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          affine.store %3, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
  func.func @_Z4tanhPA28_A28_fS1_(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>) {
    %0 = memref.get_global @FM_buffer_1 : memref<64x28x28xf16>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          affine.store %4, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    %1 = memref.get_global @FM_buffer_2 : memref<64x28x28xf16>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %5 = math.exp %4 : f16
          affine.store %5, %1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    %2 = memref.get_global @FM_buffer_3 : memref<64x28x28xf16>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %5 = arith.negf %4 : f16
          affine.store %5, %2[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %2[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %5 = math.exp %4 : f16
          affine.store %5, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %5 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %6 = arith.addf %4, %5 : f16
          affine.store %6, %2[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    %3 = memref.get_global @FM_buffer_4 : memref<64x28x28xf16>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %5 = arith.negf %4 : f16
          affine.store %5, %3[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %5 = affine.load %3[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %6 = arith.addf %4, %5 : f16
          affine.store %6, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %5 = affine.load %2[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          %6 = arith.divf %4, %5 : f16
          affine.store %6, %1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
          affine.store %4, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
  func.func @_Z3eluPA28_A28_fS1_f(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>, %arg2: f16) {
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant -1.000000e+00 : f16
    %0 = memref.get_global @FM_buffer_1 : memref<64x28x28xf16>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          affine.store %4, %0[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
        }
      }
    }
    %1 = memref.get_global @FM_buffer_2 : memref<64x28x28xf16>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %0[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %5 = math.exp %4 : f16
          affine.store %5, %1[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
        }
      }
    }
    %2 = memref.get_global @FM_buffer_3 : memref<64x28x28xf16>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          affine.store %cst_0, %2[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
        }
      }
    }
    %3 = memref.get_global @FM_buffer_4 : memref<64x28x28xf16>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %1[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %5 = affine.load %2[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %6 = arith.addf %4, %5 : f16
          affine.store %6, %3[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          affine.store %arg2, %2[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %2[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %5 = affine.load %3[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %6 = arith.mulf %4, %5 : f16
          affine.store %6, %1[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %0[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          %5 = arith.cmpf oge, %4, %cst : f16
          scf.if %5 {
            affine.store %4, %2[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          } else {
            %6 = affine.load %1[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
            affine.store %6, %2[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          }
        }
      }
    }
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %2[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
          affine.store %4, %arg1[%arg3, %arg4, %arg5] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
  func.func @_Z3topPA28_A28_fS1_S1_S1_S1_S1_(%arg0: memref<64x28x28xf16>, %arg1: memref<64x28x28xf16>, %arg2: memref<64x28x28xf16>, %arg3: memref<64x28x28xf16>, %arg4: memref<64x28x28xf16>, %arg5: memref<64x28x28xf16>) attributes {hls.allocation = [{instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}]} {
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 5.000000e-01 : f16
    %cst_1 = arith.constant -1.000000e+00 : f16
    %cst_2 = arith.constant 1.000000e+00 : f16
    %0 = memref.get_global @FM_buffer_1 : memref<64x28x28xf16>
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %arg0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          affine.store %4, %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    %1 = memref.get_global @FM_buffer_2 : memref<64x28x28xf16>
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = arith.negf %4 : f16
          affine.store %5, %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = math.exp %4 : f16
          affine.store %5, %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    %2 = memref.get_global @FM_buffer_3 : memref<64x28x28xf16>
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          affine.store %cst_2, %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %6 = arith.addf %4, %5 : f16
          affine.store %6, %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = affine.load %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %6 = arith.divf %4, %5 : f16
          affine.store %6, %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          affine.store %4, %arg1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %arg2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          affine.store %4, %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = math.exp %4 : f16
          affine.store %5, %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = arith.negf %4 : f16
          affine.store %5, %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = math.exp %4 : f16
          affine.store %5, %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %6 = arith.addf %4, %5 : f16
          affine.store %6, %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    %3 = memref.get_global @FM_buffer_4 : memref<64x28x28xf16>
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = arith.negf %4 : f16
          affine.store %5, %3[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = affine.load %3[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %6 = arith.addf %4, %5 : f16
          affine.store %6, %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = affine.load %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %6 = arith.divf %4, %5 : f16
          affine.store %6, %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          affine.store %4, %arg3[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %arg4[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          affine.store %4, %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = math.exp %4 : f16
          affine.store %5, %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          affine.store %cst_1, %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = affine.load %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %6 = arith.addf %4, %5 : f16
          affine.store %6, %3[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          affine.store %cst_0, %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = affine.load %3[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %6 = arith.mulf %4, %5 : f16
          affine.store %6, %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          %5 = arith.cmpf oge, %4, %cst : f16
          scf.if %5 {
            affine.store %4, %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          } else {
            %6 = affine.load %1[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
            affine.store %6, %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          }
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %2[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
          affine.store %4, %arg5[%arg6, %arg7, %arg8] : memref<64x28x28xf16>
        }
      }
    }
    return
  }
}
