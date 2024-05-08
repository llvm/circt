##===- CMakeLists.txt - build definitions for Sim -------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(CIRCTSim
  SimDialect.cpp
  SimOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Sim

  DEPENDS
  CIRCTHW
  CIRCTSV
  MLIRSimIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSeq
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRTransforms
)
