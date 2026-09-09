module {
  memref.global @FM_buffer_4 : memref<64x28x28xf32> = uninitialized
  memref.global @FM_buffer_3 : memref<64x28x28xf32> = uninitialized
  memref.global @FM_buffer_2 : memref<64x28x28xf32> = uninitialized
  memref.global @FM_buffer_1 : memref<64x28x28xf32> = uninitialized
  func.func @_Z16load_feature_mapPA28_A28_fS1_(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          affine.store %0, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
  func.func @_Z17store_feature_mapPA28_A28_fS1_(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          affine.store %0, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
  func.func @_Z11compute_expPA28_A28_fS1_(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %1 = math.exp %0 : f32
          affine.store %1, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
  func.func @_Z8negativePA28_A28_fS1_(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %1 = arith.negf %0 : f32
          affine.store %1, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
  func.func @_Z11compute_divPA28_A28_fS1_S1_(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>, %arg2: memref<64x28x28xf32>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %0 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %1 = affine.load %arg1[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %2 = arith.divf %0, %1 : f32
          affine.store %2, %arg2[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
  func.func @_Z11compute_addPA28_A28_fS1_S1_(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>, %arg2: memref<64x28x28xf32>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %0 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %1 = affine.load %arg1[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %2 = arith.addf %0, %1 : f32
          affine.store %2, %arg2[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
  func.func @_Z11compute_mulPA28_A28_fS1_S1_(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>, %arg2: memref<64x28x28xf32>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %0 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %1 = affine.load %arg1[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %2 = arith.mulf %0, %1 : f32
          affine.store %2, %arg2[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
  func.func @_Z9set_valuePA28_A28_ff(%arg0: memref<64x28x28xf32>, %arg1: f32) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          affine.store %arg1, %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
  func.func @_Z6selectPA28_A28_fS1_S1_f(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>, %arg2: memref<64x28x28xf32>, %arg3: f32) {
    affine.for %arg4 = 0 to 64 {
      affine.for %arg5 = 0 to 28 {
        affine.for %arg6 = 0 to 28 {
          %0 = affine.load %arg0[%arg4, %arg5, %arg6] : memref<64x28x28xf32>
          %1 = arith.cmpf oge, %0, %arg3 : f32
          scf.if %1 {
            %2 = affine.load %arg0[%arg4, %arg5, %arg6] : memref<64x28x28xf32>
            affine.store %2, %arg2[%arg4, %arg5, %arg6] : memref<64x28x28xf32>
          } else {
            %2 = affine.load %arg1[%arg4, %arg5, %arg6] : memref<64x28x28xf32>
            affine.store %2, %arg2[%arg4, %arg5, %arg6] : memref<64x28x28xf32>
          }
        }
      }
    }
    return
  }
  func.func @_Z7sigmoidPA28_A28_fS1_(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>) {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = memref.get_global @FM_buffer_1 : memref<64x28x28xf32>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          affine.store %3, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    %1 = memref.get_global @FM_buffer_2 : memref<64x28x28xf32>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %4 = arith.negf %3 : f32
          affine.store %4, %1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %4 = math.exp %3 : f32
          affine.store %4, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    %2 = memref.get_global @FM_buffer_3 : memref<64x28x28xf32>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          affine.store %cst, %2[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %2[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %4 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %5 = arith.addf %3, %4 : f32
          affine.store %5, %1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %2[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %4 = affine.load %1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %5 = arith.divf %3, %4 : f32
          affine.store %5, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %3 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          affine.store %3, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
  func.func @_Z4tanhPA28_A28_fS1_(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>) {
    %0 = memref.get_global @FM_buffer_1 : memref<64x28x28xf32>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          affine.store %4, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    %1 = memref.get_global @FM_buffer_2 : memref<64x28x28xf32>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %5 = math.exp %4 : f32
          affine.store %5, %1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    %2 = memref.get_global @FM_buffer_3 : memref<64x28x28xf32>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %5 = arith.negf %4 : f32
          affine.store %5, %2[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %2[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %5 = math.exp %4 : f32
          affine.store %5, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %5 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %6 = arith.addf %4, %5 : f32
          affine.store %6, %2[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    %3 = memref.get_global @FM_buffer_4 : memref<64x28x28xf32>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %5 = arith.negf %4 : f32
          affine.store %5, %3[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %5 = affine.load %3[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %6 = arith.addf %4, %5 : f32
          affine.store %6, %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %0[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %5 = affine.load %2[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          %6 = arith.divf %4, %5 : f32
          affine.store %6, %1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 28 {
        affine.for %arg4 = 0 to 28 {
          %4 = affine.load %1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
          affine.store %4, %arg1[%arg2, %arg3, %arg4] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
  func.func @_Z3eluPA28_A28_fS1_f(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>, %arg2: f32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -1.000000e+00 : f32
    %0 = memref.get_global @FM_buffer_1 : memref<64x28x28xf32>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          affine.store %4, %0[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
        }
      }
    }
    %1 = memref.get_global @FM_buffer_2 : memref<64x28x28xf32>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %0[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %5 = math.exp %4 : f32
          affine.store %5, %1[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
        }
      }
    }
    %2 = memref.get_global @FM_buffer_3 : memref<64x28x28xf32>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          affine.store %cst_0, %2[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
        }
      }
    }
    %3 = memref.get_global @FM_buffer_4 : memref<64x28x28xf32>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %1[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %5 = affine.load %2[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %6 = arith.addf %4, %5 : f32
          affine.store %6, %3[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          affine.store %arg2, %2[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %2[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %5 = affine.load %3[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %6 = arith.mulf %4, %5 : f32
          affine.store %6, %1[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %0[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          %5 = arith.cmpf oge, %4, %cst : f32
          scf.if %5 {
            affine.store %4, %2[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          } else {
            %6 = affine.load %1[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
            affine.store %6, %2[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          }
        }
      }
    }
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %4 = affine.load %2[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
          affine.store %4, %arg1[%arg3, %arg4, %arg5] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
  func.func @_Z3topPA28_A28_fS1_S1_S1_S1_S1_(%arg0: memref<64x28x28xf32>, %arg1: memref<64x28x28xf32>, %arg2: memref<64x28x28xf32>, %arg3: memref<64x28x28xf32>, %arg4: memref<64x28x28xf32>, %arg5: memref<64x28x28xf32>) attributes {hls.allocation = [{instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}, {instances = "", limit = 1 : i32, type = "function"}]} {
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant -1.000000e+00 : f32
    %0 = memref.get_global @FM_buffer_1 : memref<64x28x28xf32>
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %arg4[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
          affine.store %4, %0[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
        }
      }
    }
    %1 = memref.get_global @FM_buffer_2 : memref<64x28x28xf32>
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
          %5 = math.exp %4 : f32
          affine.store %5, %1[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
        }
      }
    }
    %2 = memref.get_global @FM_buffer_3 : memref<64x28x28xf32>
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          affine.store %cst_1, %2[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
        }
      }
    }
    %3 = memref.get_global @FM_buffer_4 : memref<64x28x28xf32>
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %1[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
          %5 = affine.load %2[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
          %6 = arith.addf %4, %5 : f32
          affine.store %6, %3[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          affine.store %cst, %2[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %2[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
          %5 = affine.load %3[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
          %6 = arith.mulf %4, %5 : f32
          affine.store %6, %1[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %0[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
          %5 = arith.cmpf oge, %4, %cst_0 : f32
          scf.if %5 {
            affine.store %4, %2[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
          } else {
            %6 = affine.load %1[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
            affine.store %6, %2[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
          }
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 28 {
        affine.for %arg8 = 0 to 28 {
          %4 = affine.load %2[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
          affine.store %4, %arg5[%arg6, %arg7, %arg8] : memref<64x28x28xf32>
        }
      }
    }
    return
  }
}
