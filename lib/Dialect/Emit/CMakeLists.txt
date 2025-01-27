##===- CMakeLists.txt - Emit dialect build definitions --------------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(CIRCTEmit
  EmitDialect.cpp
  EmitOpInterfaces.cpp
  EmitOps.cpp

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

add_subdirectory(Transforms)
