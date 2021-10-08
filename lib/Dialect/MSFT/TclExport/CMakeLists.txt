##===- CMakeLists.txt - build def for MSFT tcl export ---------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_translation_library(CIRCTMSFTExportTcl
  ExportQuartusTcl.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  CIRCTMSFT
  CIRCTHW
  MLIRTranslation
  )
