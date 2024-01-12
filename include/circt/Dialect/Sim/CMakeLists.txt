##===- CMakeLists.txt - Sim dialect build definitions ---------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
##
##===----------------------------------------------------------------------===//

add_circt_dialect(Sim sim)
add_circt_dialect_doc(Sim sim)
add_dependencies(circt-headers MLIRSimIncGen)
