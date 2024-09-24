add_subdirectory(IR)

##===- CMakeLists.txt - Object Model dialect build definitions ------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(
  CIRCTRTG

  ExportElf.cpp

  IR/RTGDialect.cpp
  IR/RTGTypes.cpp
  IR/RTGOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/RTG/IR

  DEPENDS
  MLIROMIncGen
  CIRCTRTGInterfacesIncGen
  CIRCTRTGTransformsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PRIVATE
  MLIRIR
  MLIRBytecodeOpInterface
)

