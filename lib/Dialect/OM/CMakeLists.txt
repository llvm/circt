##===- CMakeLists.txt - Object Model dialect build definitions ------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(
  CIRCTOM

  OMDialect.cpp
  OMOps.cpp

  DEPENDS
  MLIROMIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PRIVATE
  MLIRIR
)

add_dependencies(circt-headers MLIROMIncGen)
