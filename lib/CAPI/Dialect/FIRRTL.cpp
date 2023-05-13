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

bool firrtlTypeIs(MlirType type) { return unwrap(type).isa<FIRRTLType>(); }

bool firrtlTypeIsGround(MlirType type) {
  return unwrap(type).cast<FIRRTLType>().isGround();
}

bool firrtlTypeIsConst(MlirType type) {
  return unwrap(type).cast<FIRRTLType>().isConst();
}

bool firrtlTypeIsClock(MlirType type) { return unwrap(type).isa<ClockType>(); }

bool firrtlTypeIsReset(MlirType type) { return unwrap(type).isa<ResetType>(); }

bool firrtlTypeIsAsyncReset(MlirType type) {
  return unwrap(type).isa<AsyncResetType>();
}

bool firrtlTypeIsSInt(MlirType type) { return unwrap(type).isa<SIntType>(); }

bool firrtlTypeIsUInt(MlirType type) { return unwrap(type).isa<UIntType>(); }

bool firrtlTypeIsAnalog(MlirType type) {
  return unwrap(type).isa<AnalogType>();
}

bool firrtlTypeIsBundle(MlirType type) {
  return unwrap(type).isa<BundleType>();
}

bool firrtlTypeIsFVector(MlirType type) {
  return unwrap(type).isa<FVectorType>();
}

bool firrtlTypeIsOpenBundle(MlirType type) {
  return unwrap(type).isa<OpenBundleType>();
}

bool firrtlTypeIsOpenVector(MlirType type) {
  return unwrap(type).isa<OpenVectorType>();
}

bool firrtlTypeIsFEnum(MlirType type) { return unwrap(type).isa<FEnumType>(); }

bool firrtlTypeIsRef(MlirType type) { return unwrap(type).isa<RefType>(); }

bool firrtlTypesAreEquivalent(MlirType dest, MlirType src,
                              bool srcOuterTypeIsConst) {
  return circt::firrtl::areTypesEquivalent(unwrap(dest).cast<FIRRTLType>(),
                                           unwrap(src).cast<FIRRTLType>(),
                                           srcOuterTypeIsConst);
}

int32_t firrtlTypeGetBitWidth(MlirType type, bool ignoreFlip) {
  return getBitWidth(unwrap(type).cast<FIRRTLBaseType>(), ignoreFlip)
      .value_or(0);
}

bool firrtlTypeIsLarger(MlirType dst, MlirType src) {
  return circt::firrtl::isTypeLarger(unwrap(dst).cast<FIRRTLBaseType>(),
                                     unwrap(src).cast<FIRRTLBaseType>());
}

int32_t firrtlTypeBundleGetNumFields(MlirType type) {
  return unwrap(type).cast<BundleType>().getNumElements();
}

int32_t firrtlTypeBundleGetElementIndex(MlirType type, MlirStringRef name) {
  return unwrap(type)
      .cast<BundleType>()
      .getElementIndex(unwrap(name))
      .value_or(-1);
}

MlirStringRef firrtlTypeBundleGetElementName(MlirType type, int32_t index) {
  return wrap(unwrap(type).cast<BundleType>().getElementName(index));
}

FirrtlBundleElement firrtlTypeBundleGetElementByIndex(MlirType type,
                                                      int32_t index) {
  FirrtlBundleElement ret;
  auto bundle = unwrap(type).cast<BundleType>();
  auto field = bundle.getElement(index);
  ret.name = wrap(field.name);
  ret.type = wrap(field.type);
  ret.isFlip = field.isFlip;
  return ret;
}

FirrtlBundleElement firrtlTypeBundleGetElementByName(MlirType type,
                                                     MlirStringRef name) {
  FirrtlBundleElement ret;
  auto bundle = unwrap(type).cast<BundleType>();
  if (auto field = bundle.getElement(unwrap(name))) {
    ret.name = wrap(field->name);
    ret.type = wrap(field->type);
    ret.isFlip = field->isFlip;
  } else {
    ret.name = {nullptr};
    ret.type = {nullptr};
  }
  return ret;
}

int32_t firrtlTypeVectorGetNumFields(MlirType type) {
  return unwrap(type).cast<FVectorType>().getNumElements();
}
