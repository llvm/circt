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
MlirType mooreRealTypeGet(MlirContext ctx, unsigned width) {
  if (width == 32)
    return wrap(RealType::get(unwrap(ctx), RealWidth::f32));
  if (width == 64)
    return wrap(RealType::get(unwrap(ctx), RealWidth::f32));
  return {};
}

MlirType mooreOpenArrayTypeGet(MlirType elementType) {
  return wrap(OpenArrayType::get(cast<PackedType>(unwrap(elementType))));
}

MlirType mooreArrayTypeGet(unsigned size, MlirType elementType) {
  return wrap(ArrayType::get(size, cast<PackedType>(unwrap(elementType))));
}

MlirType mooreOpenUnpackedArrayTypeGet(MlirType elementType) {
  return wrap(
      OpenUnpackedArrayType::get(cast<UnpackedType>(unwrap(elementType))));
}

MlirType mooreUnpackedArrayTypeGet(unsigned size, MlirType elementType) {
  return wrap(
      UnpackedArrayType::get(size, cast<UnpackedType>(unwrap(elementType))));
}

MlirType mooreAssocArrayTypeGet(MlirType elementType, MlirType indexType) {
  return wrap(AssocArrayType::get(cast<UnpackedType>(unwrap(elementType)),
                                  cast<UnpackedType>(unwrap(indexType))));
}

MlirType mooreQueueTypeGet(MlirType elementType, unsigned bound) {
  return wrap(QueueType::get(cast<UnpackedType>(unwrap(elementType)), bound));
}

bool mooreIsTwoValuedType(MlirType type) {
  return cast<UnpackedType>(unwrap(type)).getDomain() == Domain::TwoValued;
}

bool mooreIsFourValuedType(MlirType type) {
  return cast<UnpackedType>(unwrap(type)).getDomain() == Domain::FourValued;
}
