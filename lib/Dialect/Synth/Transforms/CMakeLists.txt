##===----------------------------------------------------------------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(CIRCTSynthTransforms
  CutRewriter.cpp
  SynthesisPipeline.cpp
  TechMapper.cpp
  TestPriorityCuts.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Synth

  DEPENDS
  CIRCTSynthTransformsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  CIRCTSV
  CIRCTAIGTransforms
  CIRCTCombToAIG
  CIRCTCombToDatapath
  CIRCTDatapathToComb
  CIRCTHWTransforms
  CIRCTSupport
  CIRCTTransforms
  MLIRIR
  MLIRPass
  MLIRTransforms
)
