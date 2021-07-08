##===- CMakeLists.txt - Seq dialect build definitions ---------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
##
##===----------------------------------------------------------------------===//

add_circt_dialect(Seq seq)
add_circt_doc(Seq -gen-dialect-doc Seq Dialects/)

set(LLVM_TARGET_DEFINITIONS Seq.td)
mlir_tablegen(SeqPasses.h.inc -gen-pass-decls)
mlir_tablegen(SeqEnums.h.inc -gen-enum-decls)
mlir_tablegen(SeqEnums.cpp.inc -gen-enum-defs)

add_circt_doc(Seq -gen-pass-doc SeqPasses ./)

add_public_tablegen_target(MLIRSeqMiscIncGen)
