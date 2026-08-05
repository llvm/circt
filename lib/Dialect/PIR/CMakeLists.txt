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
