//===- LTL.cpp - C interface for the LTL dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/LTL.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LTL/LTLTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

using namespace circt;
using namespace circt::ltl;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(LTL, ltl, circt::ltl::LTLDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

bool ltlTypeIsASequence(MlirType type) {
  return isa<SequenceType>(unwrap(type));
}

MlirType ltlSequenceTypeGet(MlirContext ctx) {
  return wrap(SequenceType::get(unwrap(ctx)));
}

bool ltlTypeIsAProperty(MlirType type) {
  return isa<PropertyType>(unwrap(type));
}

MlirType ltlPropertyTypeGet(MlirContext ctx) {
  return wrap(PropertyType::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

static_assert(LTL_CLOCK_EDGE_POS == static_cast<unsigned>(ClockEdge::Pos));
static_assert(LTL_CLOCK_EDGE_NEG == static_cast<unsigned>(ClockEdge::Neg));
static_assert(LTL_CLOCK_EDGE_BOTH == static_cast<unsigned>(ClockEdge::Both));

bool ltlAttrIsAClockEdgeAttr(MlirAttribute attr) {
  return isa<ClockEdgeAttr>(unwrap(attr));
}

MlirAttribute ltlClockEdgeAttrGet(MlirContext ctx, LTLClockEdge edge) {
  return wrap(ClockEdgeAttr::get(unwrap(ctx), static_cast<ClockEdge>(edge)));
}

LTLClockEdge ltlClockEdgeAttrGetValue(MlirAttribute attr) {
  auto edge = cast<ClockEdgeAttr>(unwrap(attr)).getValue();
  return static_cast<LTLClockEdge>(static_cast<unsigned>(edge));
}
