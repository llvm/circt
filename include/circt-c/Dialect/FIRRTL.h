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
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Return 'true' if this is any FIRRTL type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsFIRRTLType(MlirType type);

/// Return 'true' if this is a FIRRTL ground type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsGround(MlirType type);

/// Return 'true' if this is a FIRRTL const type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsConst(MlirType type);

/// Return 'true' if this is a FIRRTL Clock type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsClock(MlirType type);

/// Return 'true' if this is a FIRRTL Reset type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsReset(MlirType type);

/// Return 'true' if this is a FIRRTL AsyncReset type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAsyncReset(MlirType type);

/// Return 'true' if this is a FIRRTL SInt type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsSInt(MlirType type);

/// Return 'true' if this is a FIRRTL UInt type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsUInt(MlirType type);

/// Return 'true' if this is a FIRRTL Analog type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAnalog(MlirType type);

/// Return 'true' if this is a FIRRTL Bundle type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsBundle(MlirType type);

/// Return 'true' if this is a FIRRTL Vector type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsFVector(MlirType type);

/// Return 'true' if this is a FIRRTL Reference type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsRef(MlirType type);

/// Return 'true' if this is a FIRRTL OpenBundle type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsOpenBundle(MlirType type);

/// Return 'true' if this is a FIRRTL OpenVector type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsOpenVector(MlirType type);

/// Return 'true' if this is a FIRRTL Enum type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsFEnum(MlirType type);

/// Return the bit-width of a type.
MLIR_CAPI_EXPORTED int32_t firrtlTypeGetBitWidth(MlirType type,
                                                 bool ignoreFlip);

/// Return 'true' if the destination type is at least as wide as the source.
MLIR_CAPI_EXPORTED bool firrtlTypeIsLarger(MlirType dst, MlirType src);

/// Return 'true' if two types are equivalent.
MLIR_CAPI_EXPORTED bool firrtlTypesAreEquivalent(MlirType dest, MlirType src,
                                                 bool srcOuterTypeIsConst);

//===----------------------------------------------------------------------===//
// Aggregate Types
//===----------------------------------------------------------------------===//

/// Wrapped version of BundleType::BundleElement
typedef struct {
  MlirIdentifier name;
  bool isFlip;
  MlirType type;
} FirrtlBundleField;

/// Return the bundle field at the provided index.
MLIR_CAPI_EXPORTED FirrtlBundleField
firrtlTypeBundleGetFieldByIndex(MlirType type, int32_t index);

/// Return the bundle field with the provided name.
MLIR_CAPI_EXPORTED FirrtlBundleField
firrtlTypeBundleGetFieldByName(MlirType type, MlirStringRef name);

/// Return the number of fields for the provided bundle type.
MLIR_CAPI_EXPORTED int32_t firrtlTypeBundleGetNumFields(MlirType type);

/// Return the index of the bundle field with the provided name.
/// Returns (-1) if the field does not exist.
MLIR_CAPI_EXPORTED int32_t firrtlTypeBundleGetFieldIndex(MlirType type,
                                                         MlirStringRef name);

/// Returns 'true' if a bundle field exists with the provided name.
MLIR_CAPI_EXPORTED bool firrtlTypeBundleHasFieldName(MlirType type,
                                                     MlirStringRef name);

/// Return the name of the bundle field at the provided index.
MLIR_CAPI_EXPORTED MlirStringRef firrtlTypeBundleGetFieldName(MlirType type,
                                                              int32_t index);

/// Return the number of elements for the provided vector type.
MLIR_CAPI_EXPORTED int32_t firrtlTypeVectorGetNumElements(MlirType type);

/// Return the type of all elements in the provided vector type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeVectorGetElementType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_FIRRTL_H
