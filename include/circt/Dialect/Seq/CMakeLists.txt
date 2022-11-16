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
add_circt_dialect_doc(Seq seq)

set(LLVM_TARGET_DEFINITIONS SeqPasses.td)
mlir_tablegen(SeqPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTSeqTransformsIncGen)
add_circt_doc(SeqPasses SeqPasses -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS SeqOpInterfaces.td)
mlir_tablegen(SeqOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SeqOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTSeqOpInterfacesIncGen)
add_dependencies(circt-headers CIRCTSeqOpInterfacesIncGen)
