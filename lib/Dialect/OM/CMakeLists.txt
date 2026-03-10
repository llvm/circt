##===- CMakeLists.txt - Object Model dialect build definitions ------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

set(CIRCT_OM_Sources
  OMAttributes.cpp
  OMDialect.cpp
  OMOpInterfaces.cpp
  OMOps.cpp
  OMTypes.cpp
  OMUtils.cpp
)

set(LLVM_OPTIONAL_SOURCES
  ${CIRCT_OM_Sources}
  OMReductions.cpp
)

add_circt_dialect_library(
  CIRCTOM
  ${CIRCT_OM_Sources}

  DEPENDS
  MLIROMIncGen
  MLIROMAttrIncGen
  MLIROMOpInterfacesIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PRIVATE
  CIRCTHW
  MLIRIR
)

add_circt_library(CIRCTOMReductions
  OMReductions.cpp

  LINK_LIBS PUBLIC
  CIRCTReduceLib
  CIRCTOM
  MLIRIR
)

add_dependencies(circt-headers MLIROMIncGen)

add_subdirectory(Evaluator)
add_subdirectory(Transforms)
