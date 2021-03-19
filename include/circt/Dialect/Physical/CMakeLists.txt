##===- CMakeLists.txt - Physical dialect build definitions ----*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
##
##===----------------------------------------------------------------------===//

add_circt_dialect(Physical physical)
add_circt_doc(Physical -gen-dialect-doc Physical Dialects/)

set(LLVM_TARGET_DEFINITIONS Physical.td)
