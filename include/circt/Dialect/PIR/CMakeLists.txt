##===- CMakeLists.txt - Property IR dialect build definitions ------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect(PIR pir)
add_circt_doc(PIROps Dialects/PIROps -gen-op-doc)
add_circt_doc(PIRTypes Dialects/PIRTypes -gen-typedef-doc -dialect pir)

set(LLVM_TARGET_DEFINITIONS PIREnums.td)
mlir_tablegen(PIREnums.h.inc -gen-enum-decls)
mlir_tablegen(PIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTPIREnumsIncGen)
add_dependencies(circt-headers CIRCTPIREnumsIncGen)

set(LLVM_TARGET_DEFINITIONS PIROps.td)
mlir_tablegen(PIRAttrDefs.h.inc -gen-attrdef-decls
-attrdefs-dialect=pir)
mlir_tablegen(PIRAttrDefs.cpp.inc -gen-attrdef-defs
-attrdefs-dialect=pir)
add_public_tablegen_target(CIRCTPIRAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS PIROps.td)
mlir_tablegen(PIRInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(PIRInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTPIRInterfaceDefsIncGen)
