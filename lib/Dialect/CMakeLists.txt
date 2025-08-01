##===- CMakeLists.txt - ---------------------------------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
##
##===----------------------------------------------------------------------===//

add_subdirectory(AIG)
add_subdirectory(Arc)
add_subdirectory(Calyx)
add_subdirectory(Comb)
add_subdirectory(Datapath)
add_subdirectory(DC)
add_subdirectory(Debug)
add_subdirectory(Emit)
add_subdirectory(ESI)
add_subdirectory(FIRRTL)
add_subdirectory(FSM)
add_subdirectory(Handshake)
add_subdirectory(HW)
add_subdirectory(HWArith)
add_subdirectory(Kanagawa)
add_subdirectory(Interop)
add_subdirectory(LLHD)
add_subdirectory(LoopSchedule)
add_subdirectory(LTL)
add_subdirectory(Moore)
add_subdirectory(MSFT)
add_subdirectory(OM)
add_subdirectory(Pipeline)
add_subdirectory(RTG)
if (CIRCT_INCLUDE_TESTS)
  add_subdirectory(RTGTest)
endif()
add_subdirectory(Seq)
add_subdirectory(Sim)
add_subdirectory(SSP)
add_subdirectory(SV)
add_subdirectory(Synth)
add_subdirectory(SystemC)
add_subdirectory(Verif)
