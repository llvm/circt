##===- CMakeLists.txt - MSFT dialect build definitions --------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect(MSFT msft)
add_circt_doc(MSFT -gen-dialect-doc MSFT Dialects/)

set(LLVM_TARGET_DEFINITIONS MSFT.td)
