//===- LTL.h - C interface for the LTL dialect --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_LTL_H
#define CIRCT_C_DIALECT_LTL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTNEXTLINE(modernize-use-using)
typedef enum LTLClockEdge {
  LTL_CLOCK_EDGE_POS = 0,
  LTL_CLOCK_EDGE_NEG = 1,
  LTL_CLOCK_EDGE_BOTH = 2,
} LTLClockEdge;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(LTL, ltl);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool ltlTypeIsASequence(MlirType type);
MLIR_CAPI_EXPORTED MlirType ltlSequenceTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED bool ltlTypeIsAProperty(MlirType type);
MLIR_CAPI_EXPORTED MlirType ltlPropertyTypeGet(MlirContext ctx);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool ltlAttrIsAClockEdgeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute ltlClockEdgeAttrGet(MlirContext ctx,
                                                     LTLClockEdge edge);
MLIR_CAPI_EXPORTED LTLClockEdge ltlClockEdgeAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_LTL_H
