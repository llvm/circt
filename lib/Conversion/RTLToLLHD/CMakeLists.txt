# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_circt_conversion_library(MLIRRTLToLLHD
  RTLToLLHD.cpp

  DEPENDS
  MLIRRTLToLLHDConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRLLHD
  MLIRRTL
  MLIRTransforms
)
