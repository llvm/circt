##===- CMakeLists.txt - Emit dialect build definitions --------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect(Emit emit)
add_circt_dialect_doc(Emit emit)
add_dependencies(circt-headers MLIREmitIncGen)

set(LLVM_TARGET_DEFINITIONS EmitOpInterfaces.td)
mlir_tablegen(EmitOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(EmitOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTEmitOpInterfacesIncGen)
add_dependencies(circt-headers CIRCTEmitOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS EmitPasses.td)
mlir_tablegen(EmitPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTEmitTransformsIncGen)
add_circt_doc(EmitPasses EmitPasses -gen-pass-doc)
