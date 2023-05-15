//===- FIRRTL.cpp - C Interface for the FIRRTL Dialect --------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/FIRRTL.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl,
                                      circt::firrtl::FIRRTLDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool firrtlTypeIsFIRRTLType(MlirType type) {
  return unwrap(type).isa<FIRRTLType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsGround(MlirType type) {
  return unwrap(type).cast<FIRRTLType>().isGround();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsConst(MlirType type) {
  return unwrap(type).cast<FIRRTLType>().isConst();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsClock(MlirType type) {
  return unwrap(type).isa<ClockType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsReset(MlirType type) {
  return unwrap(type).isa<ResetType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsAsyncReset(MlirType type) {
  return unwrap(type).isa<AsyncResetType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsSInt(MlirType type) {
  return unwrap(type).isa<SIntType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsUInt(MlirType type) {
  return unwrap(type).isa<UIntType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsAnalog(MlirType type) {
  return unwrap(type).isa<AnalogType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsBundle(MlirType type) {
  return unwrap(type).isa<BundleType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsFVector(MlirType type) {
  return unwrap(type).isa<FVectorType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsOpenBundle(MlirType type) {
  return unwrap(type).isa<OpenBundleType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsOpenVector(MlirType type) {
  return unwrap(type).isa<OpenVectorType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsFEnum(MlirType type) {
  return unwrap(type).isa<FEnumType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsRef(MlirType type) {
  return unwrap(type).isa<RefType>();
}

MLIR_CAPI_EXPORTED bool firrtlTypesAreEquivalent(MlirType dest, MlirType src,
                                                 bool srcOuterTypeIsConst) {
  return circt::firrtl::areTypesEquivalent(unwrap(dest).cast<FIRRTLType>(),
                                           unwrap(src).cast<FIRRTLType>(),
                                           srcOuterTypeIsConst);
}

MLIR_CAPI_EXPORTED int32_t firrtlTypeGetBitWidth(MlirType type,
                                                 bool ignoreFlip) {
  return getBitWidth(unwrap(type).cast<FIRRTLBaseType>(), ignoreFlip)
      .value_or(0);
}

MLIR_CAPI_EXPORTED bool firrtlTypeIsLarger(MlirType dst, MlirType src) {
  return circt::firrtl::isTypeLarger(unwrap(dst).cast<FIRRTLBaseType>(),
                                     unwrap(src).cast<FIRRTLBaseType>());
}

//===----------------------------------------------------------------------===//
// Aggregate Types
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED FirrtlBundleField
firrtlTypeBundleGetFieldByIndex(MlirType type, int32_t index) {
  FirrtlBundleField ret;
  auto bundle = unwrap(type).cast<BundleType>();
  auto field = bundle.getElement(index);
  ret.name = wrap(field.name);
  ret.type = wrap(field.type);
  ret.isFlip = field.isFlip;
  return ret;
}

MLIR_CAPI_EXPORTED FirrtlBundleField
firrtlTypeBundleGetFieldByName(MlirType type, MlirStringRef name) {
  FirrtlBundleField ret;
  auto bundle = unwrap(type).cast<BundleType>();
  auto field = bundle.getElement(unwrap(name)).value();
  ret.name = wrap(field.name);
  ret.type = wrap(field.type);
  ret.isFlip = field.isFlip;
  return ret;
}

MLIR_CAPI_EXPORTED int32_t firrtlTypeBundleGetNumFields(MlirType type) {
  return unwrap(type).cast<BundleType>().getNumElements();
}

MLIR_CAPI_EXPORTED int32_t firrtlTypeBundleGetFieldIndex(MlirType type,
                                                         MlirStringRef name) {
  return unwrap(type)
      .cast<BundleType>()
      .getElementIndex(unwrap(name))
      .value_or(-1);
}

MLIR_CAPI_EXPORTED bool firrtlTypeBundleHasFieldName(MlirType type,
                                                     MlirStringRef name) {
  auto bundle = unwrap(type).cast<BundleType>();
  auto field = bundle.getElement(unwrap(name));
  return field ? true : false;
}

MLIR_CAPI_EXPORTED MlirStringRef firrtlTypeBundleGetFieldName(MlirType type,
                                                              int32_t index) {
  return wrap(unwrap(type).cast<BundleType>().getElementName(index));
}

MLIR_CAPI_EXPORTED int32_t firrtlTypeVectorGetNumElements(MlirType type) {
  return unwrap(type).cast<FVectorType>().getNumElements();
}

MLIR_CAPI_EXPORTED MlirType firrtlTypeVectorGetElementType(MlirType type) {
  auto vector = unwrap(type).cast<FVectorType>();
  return wrap(vector.getElementType());
}
