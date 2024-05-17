//===- Moore.h - C interface for the Moore dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Moore.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt;
using namespace circt::moore;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Moore, moore, MooreDialect)

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

static RealType::Kind convertMooreRealKind(enum MooreRealKind kind) {
  switch (kind) {
  case MooreRealKind::MooreShortReal:
    return circt::moore::RealType::ShortReal;
  case MooreRealKind::MooreReal:
    return circt::moore::RealType::Real;
  case MooreRealKind::MooreRealTime:
    return circt::moore::RealType::RealTime;
  }
  llvm_unreachable("All cases should be covered.");
}

/// Create a void type.
MlirType mooreVoidTypeGet(MlirContext ctx) {
  return wrap(VoidType::get(unwrap(ctx)));
}

/// Create a string type.
MlirType mooreStringTypeGet(MlirContext ctx) {
  return wrap(StringType::get(unwrap(ctx)));
}

/// Create a chandle type.
MlirType mooreChandleTypeGet(MlirContext ctx) {
  return wrap(ChandleType::get(unwrap(ctx)));
}

/// Create a event type.
MlirType mooreEventTypeGet(MlirContext ctx) {
  return wrap(EventType::get(unwrap(ctx)));
}

/// Create a two-valued simple bit vector type.
MlirType mooreIntTypeGetInt(MlirContext ctx, unsigned width) {
  return wrap(IntType::getInt(unwrap(ctx), width));
}

/// Create a four-valued simple bit vector type.
MlirType mooreIntTypeGetLogic(MlirContext ctx, unsigned width) {
  return wrap(IntType::getLogic(unwrap(ctx), width));
}

/// Create a real type.
MlirType mooreRealTypeGet(MlirContext ctx, enum MooreRealKind kind) {
  return wrap(RealType::get(unwrap(ctx), convertMooreRealKind(kind)));
}

/// Create a packed unsized dimension type.
MlirType moorePackedUnsizedDimTypeGet(MlirType inner) {
  return wrap(PackedUnsizedDim::get(cast<PackedType>(unwrap(inner))));
}

/// Create a packed range dimension type.
MlirType moorePackedRangeDimTypeGet(MlirType inner, unsigned size, bool upDir,
                                    int offset) {
  RangeDir dir = upDir ? RangeDir::Up : RangeDir::Down;
  return wrap(
      PackedRangeDim::get(cast<PackedType>(unwrap(inner)), size, dir, offset));
}

/// Create a unpacked unsized dimension type.
MlirType mooreUnpackedUnsizedDimTypeGet(MlirType inner) {
  return wrap(UnpackedUnsizedDim::get(cast<UnpackedType>(unwrap(inner))));
}

/// Create a unpacked array dimension type.
MlirType mooreUnpackedArrayDimTypeGet(MlirType inner, unsigned size) {
  return wrap(UnpackedArrayDim::get(cast<UnpackedType>(unwrap(inner)), size));
}

/// Create a unpacked range dimension type.
MlirType mooreUnpackedRangeDimTypeGet(MlirType inner, unsigned size, bool upDir,
                                      int offset) {
  RangeDir dir = upDir ? RangeDir::Up : RangeDir::Down;
  return wrap(UnpackedRangeDim::get(cast<UnpackedType>(unwrap(inner)), size,
                                    dir, offset));
}

/// Create a unpacked assoc dimension type without index.
MlirType mooreUnpackedAssocDimTypeGet(MlirType inner) {
  return wrap(UnpackedAssocDim::get(cast<UnpackedType>(unwrap(inner))));
}

/// Create a unpacked assoc dimension type with index.
MlirType mooreUnpackedAssocDimTypeGetWithIndex(MlirType inner,
                                               MlirType indexType) {
  return wrap(UnpackedAssocDim::get(cast<UnpackedType>(unwrap(inner)),
                                    cast<UnpackedType>(unwrap(indexType))));
}

/// Create a unpacked queue dimension type without bound.
MlirType mooreUnpackedQueueDimTypeGet(MlirType inner) {
  return wrap(UnpackedQueueDim::get(cast<UnpackedType>(unwrap(inner))));
}

/// Create a unpacked queue dimension type with bound.
MlirType mooreUnpackedQueueDimTypeGetWithBound(MlirType inner, unsigned bound) {
  return wrap(UnpackedQueueDim::get(cast<UnpackedType>(unwrap(inner)), bound));
}

/// Checks whether the passed UnpackedType is a two-valued type.
bool mooreIsTwoValuedType(MlirType type) {
  return cast<UnpackedType>(unwrap(type)).getDomain() == Domain::TwoValued;
}

/// Checks whether the passed UnpackedType is a four-valued type.
bool mooreIsFourValuedType(MlirType type) {
  return cast<UnpackedType>(unwrap(type)).getDomain() == Domain::FourValued;
}
