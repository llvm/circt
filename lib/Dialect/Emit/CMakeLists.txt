##===- CMakeLists.txt - Emit dialect build definitions --------------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

set(CIRCT_EMIT_Sources
  EmitDialect.cpp
  EmitOpInterfaces.cpp
  EmitOps.cpp
)

set(LLVM_OPTIONAL_SOURCES
  ${CIRCT_EMIT_Sources}
  EmitReductions.cpp
)

add_circt_dialect_library(CIRCTEmit
  ${CIRCT_EMIT_Sources}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Emit

  DEPENDS
  MLIREmitIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRPass
  MLIRTransforms
)

add_circt_library(CIRCTEmitReductions
  EmitReductions.cpp

  LINK_LIBS PUBLIC
  CIRCTReduceLib
  CIRCTEmit
  CIRCTHW
  MLIRIR
)

add_subdirectory(Transforms)
