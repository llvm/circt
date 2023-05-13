//===-- circt-c/Dialect/FIRRTL.h - C API for FIRRTL dialect -------*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// FIRRTL dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_FIRRTL_H
#define CIRCT_C_DIALECT_FIRRTL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Wraps a FIRRTL BundleType element.
struct FirrtlBundleElement {
  MlirIdentifier name;
  bool isFlip;
  MlirType type;
};
typedef struct FirrtlBundleElement FirrtlBundleElement;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Return 'true' if this is any FIRRTL type.
bool firrtlTypeIs(MlirType type);

/// Return 'true' if this is a FIRRTL ground type.
bool firrtlTypeIsGround(MlirType type);

/// Return 'true' if this is a FIRRTL const type.
bool firrtlTypeIsConst(MlirType type);

/// Return 'true' if this is a FIRRTL Clock type.
bool firrtlTypeIsClock(MlirType type);

/// Return 'true' if this is a FIRRTL Reset type.
bool firrtlTypeIsReset(MlirType type);

/// Return 'true' if this is a FIRRTL AsyncReset type.
bool firrtlTypeIsAsyncReset(MlirType type);

/// Return 'true' if this is a FIRRTL SInt type.
bool firrtlTypeIsSInt(MlirType type);

/// Return 'true' if this is a FIRRTL UInt type.
bool firrtlTypeIsUInt(MlirType type);

/// Return 'true' if this is a FIRRTL Analog type.
bool firrtlTypeIsAnalog(MlirType type);

/// Return 'true' if this is a FIRRTL Bundle type.
bool firrtlTypeIsBundle(MlirType type);

/// Return 'true' if this is a FIRRTL Vector type.
bool firrtlTypeIsFVector(MlirType type);

/// Return 'true' if this is a FIRRTL Reference type.
bool firrtlTypeIsRef(MlirType type);

// bool firrtlTypeIsOpenBundle(MlirType type);
// bool firrtlTypeIsOpenVector(MlirType type);
// bool firrtlTypeIsFEnum(MlirType type);

/// Return the bit-width of a type.
int32_t firrtlTypeGetBitWidth(MlirType type, bool ignoreFlip);

/// Return 'true' if the destination type is at least as wide as the source.
bool firrtlTypeIsLarger(MlirType dst, MlirType src);

/// Return 'true' if two types are equivalent.
bool firrtlTypesAreEquivalent(MlirType dest, MlirType src,
                              bool srcOuterTypeIsConst);

/// Return the number of fields for the given bundle type.
int32_t firrtlTypeBundleGetNumFields(MlirType type);

/// Return the index of a named bundle field.
int32_t firrtlTypeBundleGetElementIndex(MlirType type, MlirStringRef name);

/// Return the name of the bundle field at the given index.
MlirStringRef firrtlTypeBundleGetElementName(MlirType type, int32_t index);

/// Return the bundle field at the given index.
FirrtlBundleElement firrtlTypeBundleGetElementByIndex(MlirType type,
                                                      int32_t index);

/// Return the bundle field with the given name.
FirrtlBundleElement firrtlTypeBundleGetElementByName(MlirType type,
                                                     MlirStringRef name);

/// Return the number of fields for this vector type.
int32_t firrtlTypeVectorGetNumFields(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_FIRRTL_H
