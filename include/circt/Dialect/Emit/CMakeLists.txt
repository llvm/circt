##===- CMakeLists.txt - Emit dialect build definitions --------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
##
##===----------------------------------------------------------------------===//

add_circt_dialect(Emit emit)
add_circt_dialect_doc(Emit emit)
add_dependencies(circt-headers MLIREmitIncGen)
