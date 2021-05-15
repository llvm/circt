//===-- circt-c/Dialect/HW.h - C API for HW dialect ---------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// HW dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_HW_H
#define CIRCT_C_DIALECT_HW_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Named MLIR attribute.
///
/// A named attribute is essentially a (name, attribute) pair where the name is
/// a string.

struct RTLStructFieldInfo {
  MlirStringRef name;
  MlirAttribute attribute;
};
typedef struct MlirNamedAttribute MlirNamedAttribute;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(RTL, rtl);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Return the hardware bit width of a type. Does not reflect any encoding,
/// padding, or storage scheme, just the bit (and wire width) of a
/// statically-size type. Reflects the number of wires needed to transmit a
/// value of this type. Returns -1 if the type is not known or cannot be
/// statically computed.
MLIR_CAPI_EXPORTED int64_t rtlGetBitWidth(MlirType);

/// Return true if the specified type can be used as an RTL value type, that is
/// the set of types that can be composed together to represent synthesized,
/// hardware but not marker types like InOutType or unknown types from other
/// dialects.
MLIR_CAPI_EXPORTED bool rtlTypeIsAValueType(MlirType);

/// If the type is an RTL array
MLIR_CAPI_EXPORTED bool rtlTypeIsAArrayType(MlirType);

/// If the type is an RTL inout.
MLIR_CAPI_EXPORTED bool rtlTypeIsAInOut(MlirType type);

/// If the type is an RTL struct.
MLIR_CAPI_EXPORTED bool rtlTypeIsAStructType(MlirType);

/// Creates a fixed-size RTL array type in the context associated with element
MLIR_CAPI_EXPORTED MlirType rtlArrayTypeGet(MlirType element, size_t size);

/// returns the element type of an array type
MLIR_CAPI_EXPORTED MlirType rtlArrayTypeGetElementType(MlirType);

/// returns the size of an array type
MLIR_CAPI_EXPORTED intptr_t rtlArrayTypeGetSize(MlirType);

/// Creates an RTL inout type in the context associated with element.
MLIR_CAPI_EXPORTED MlirType rtlInOutTypeGet(MlirType element);

/// Returns the element type of an inout type.
MLIR_CAPI_EXPORTED MlirType rtlInOutTypeGetElementType(MlirType);

/// Creates an RTL struct type in the context associated with the elements.
MLIR_CAPI_EXPORTED MlirType
rtlStructTypeGet(MlirContext ctx, intptr_t numElements,
                 struct RTLStructFieldInfo const *elements);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_HW_H
