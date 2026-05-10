##===----------------------------------------------------------------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect(Synth synth)
add_circt_dialect_doc(Synth synth)
add_circt_interface(SynthOpInterfaces)
add_subdirectory(Transforms)

set(LLVM_TARGET_DEFINITIONS Synth.td)

mlir_tablegen(SynthEnums.h.inc -gen-enum-decls)
mlir_tablegen(SynthEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTSynthEnumsIncGen)
add_dependencies(circt-headers CIRCTSynthEnumsIncGen)

mlir_tablegen(SynthAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=synth)
mlir_tablegen(SynthAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=synth)
add_public_tablegen_target(CIRCTSynthAttrIncGen)
add_dependencies(circt-headers CIRCTSynthAttrIncGen)
