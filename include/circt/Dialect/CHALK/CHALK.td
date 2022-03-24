//===- CHALK.td - Definition of CHALK dialect ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_TD
#define CIRCT_DIALECT_CHALK_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "circt/Types.td"

include "CHALKDialect.td"
include "CHALKAttributes.td"
include "CHALKTypes.td"
include "CHALKOps.td"

#endif // CIRCT_DIALECT_CHALK_TD
