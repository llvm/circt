# ===- CMakeLists.txt - HSDbg driver cmake --------------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//
#
# Configure and copy a script to run HSDbg visualization.
#
# ===-----------------------------------------------------------------------===//

set(SOURCES hsdbg)
foreach(file IN ITEMS ${SOURCES})
  configure_file(${file}.in ${CIRCT_TOOLS_DIR}/${file} @ONLY)
  list(APPEND OUTPUTS ${CIRCT_TOOLS_DIR}/${file})
endforeach()
add_custom_target(hsdbg-driver SOURCES ${OUTPUTS})
