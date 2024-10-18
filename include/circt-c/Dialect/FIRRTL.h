//===- FIRRTL.h - C interface for the FIRRTL dialect --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the C-API for FIRRTL dialect.
//
// A complete example of using these C-APIs is Chisel, see
// https://github.com/chipsalliance/chisel/blob/4f392323e9160440961b9f06e383d3f2742d2f3e/panamaconverter/src/PanamaCIRCTConverter.scala
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_FIRRTL_H
#define CIRCT_C_DIALECT_FIRRTL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Module instantiation conventions.
// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLConvention {
  FIRRTL_CONVENTION_INTERNAL,
  FIRRTL_CONVENTION_SCALARIZED,
} FIRRTLConvention;

/// Port direction.
// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLDirection {
  FIRRTL_DIRECTION_IN,
  FIRRTL_DIRECTION_OUT,
} FIRRTLDirection;

/// Name preservation.
///
/// Names tagged with `FIRRTL_NAME_KIND_INTERESTING_NAME` will be preserved.
// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLNameKind {
  FIRRTL_NAME_KIND_DROPPABLE_NAME,
  FIRRTL_NAME_KIND_INTERESTING_NAME,
} FIRRTLNameKind;

/// Read-Under-Write behaviour.
// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLRUW {
  FIRRTL_RUW_UNDEFINED,
  FIRRTL_RUW_OLD,
  FIRRTL_RUW_NEW,
} FIRRTLRUW;

/// Memory port direction.
// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLMemDir {
  FIRRTL_MEM_DIR_INFER,
  FIRRTL_MEM_DIR_READ,
  FIRRTL_MEM_DIR_WRITE,
  FIRRTL_MEM_DIR_READ_WRITE,
} FIRRTLMemDir;

/// Edge control trigger.
// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLEventControl {
  FIRRTL_EVENT_CONTROL_AT_POS_EDGE,
  FIRRTL_EVENT_CONTROL_AT_NEG_EDGE,
  FIRRTL_EVENT_CONTROL_AT_EDGE,
} FIRRTLEventControl;

/// Flow of value.
// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLValueFlow {
  FIRRTL_VALUE_FLOW_NONE,
  FIRRTL_VALUE_FLOW_SOURCE,
  FIRRTL_VALUE_FLOW_SINK,
  FIRRTL_VALUE_FLOW_DUPLEX,
} FIRRTLValueFlow;

/// Describes a field in a bundle type.
// NOLINTNEXTLINE(modernize-use-using)
typedef struct FIRRTLBundleField {
  MlirIdentifier name;
  bool isFlip;
  MlirType type;
} FIRRTLBundleField;

/// Describes an element in a class type.
// NOLINTNEXTLINE(modernize-use-using)
typedef struct FIRRTLClassElement {
  MlirIdentifier name;
  MlirType type;
  FIRRTLDirection direction;
} FIRRTLClassElement;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Creates a unsigned integer type with the specified width.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetUInt(MlirContext ctx, int32_t width);

/// Creates a signed integer type with the specified width.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetSInt(MlirContext ctx, int32_t width);

/// Creates a clock type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetClock(MlirContext ctx);

/// Creates a reset type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetReset(MlirContext ctx);

/// Creates a async reset type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetAsyncReset(MlirContext ctx);

/// Creates a analog type with the specified width.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetAnalog(MlirContext ctx, int32_t width);

/// Creates a vector type with the specified element type and count.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetVector(MlirContext ctx,
                                                MlirType element, size_t count);

/// Returns true if the specified type is an open bundle type.
///
/// An open bundle type means that it contains non FIRRTL base types.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAOpenBundle(MlirType type);

/// Creates a bundle type with the specified fields.
///
/// If any field has a non-FIRRTL base type, an open bundle type is returned,
/// otherwise a normal bundle type is returned.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetBundle(
    MlirContext ctx, size_t count, const FIRRTLBundleField *fields);

/// Returns the index of the field with the specified name in the bundle type.
MLIR_CAPI_EXPORTED unsigned
firrtlTypeGetBundleFieldIndex(MlirType type, MlirStringRef fieldName);

/// Creates a ref type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetRef(MlirType target, bool forceable);

/// Creates an anyref type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetAnyRef(MlirContext ctx);

/// Creates a property integer type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetInteger(MlirContext ctx);

/// Creates a property double type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetDouble(MlirContext ctx);

/// Creates a property string type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetString(MlirContext ctx);

/// Creates a property boolean type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetBoolean(MlirContext ctx);

/// Creates a property path type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetPath(MlirContext ctx);

/// Creates a property path type with the specified element type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetList(MlirContext ctx,
                                              MlirType elementType);

/// Creates a class type with the specified name and elements.
MLIR_CAPI_EXPORTED MlirType
firrtlTypeGetClass(MlirContext ctx, MlirAttribute name, size_t numberOfElements,
                   const FIRRTLClassElement *elements);

/// Returns this type with all ground types replaced with UInt<1>. This is
/// used for `mem` operations.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetMaskType(MlirType type);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

/// Creates an ConventionAttr with the specified value.
MLIR_CAPI_EXPORTED MlirAttribute
firrtlAttrGetConvention(MlirContext ctx, FIRRTLConvention convention);

/// Creates a DenseBoolArrayAttr with the specified port directions.
MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetPortDirs(
    MlirContext ctx, size_t count, const FIRRTLDirection *dirs);

/// Creates a ParamDeclAttr with the specified name, type, and value. This is
/// used for module or instance parameter definition.
MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetParamDecl(MlirContext ctx,
                                                        MlirIdentifier name,
                                                        MlirType type,
                                                        MlirAttribute value);

/// Creates a NameKindEnumAttr with the specified name preservation semantic.
MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetNameKind(MlirContext ctx,
                                                       FIRRTLNameKind nameKind);

/// Creates a RUWAttr with the specified Read-Under-Write Behaviour.
MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetRUW(MlirContext ctx,
                                                  FIRRTLRUW ruw);

/// Creates a MemoryInitAttr with the specified memory initialization
/// information.
MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetMemInit(MlirContext ctx,
                                                      MlirIdentifier filename,
                                                      bool isBinary,
                                                      bool isInline);

/// Creates a MemDirAttr with the specified memory port direction.
MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetMemDir(MlirContext ctx,
                                                     FIRRTLMemDir dir);

/// Creates a EventControlAttr with the specified value.
MLIR_CAPI_EXPORTED MlirAttribute
firrtlAttrGetEventControl(MlirContext ctx, FIRRTLEventControl eventControl);

/// Creates an IntegerAttr from a string representation of integer.
///
/// This is a workaround for supporting large integers. See
/// https://github.com/llvm/llvm-project/issues/84190#issuecomment-2035552035
MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetIntegerFromString(
    MlirType type, unsigned numBits, MlirStringRef str, uint8_t radix);

//===----------------------------------------------------------------------===//
// Utility API.
//===----------------------------------------------------------------------===//

/// Computes the flow for a Value, \p value, as determined by the FIRRTL
/// specification.  This recursively walks backwards from \p value to the
/// declaration.  The resulting flow is a combination of the declaration flow
/// (output ports and instance inputs are sinks, registers and wires are
/// duplex, anything else is a source) and the number of intermediary flips.
/// An even number of flips will result in the same flow as the declaration.
/// An odd number of flips will result in reversed flow being returned.  The
/// reverse of source is sink.  The reverse of sink is source.  The reverse of
/// duplex is duplex.  The \p flow parameter sets the initial flow.
/// A user should normally \a not have to change this from its default of \p
/// Flow::Source.
MLIR_CAPI_EXPORTED FIRRTLValueFlow firrtlValueFoldFlow(MlirValue value,
                                                       FIRRTLValueFlow flow);

/// Deserializes a JSON value into FIRRTL Annotations.  Annotations are
/// represented as a Target-keyed arrays of attributes.  The input JSON value is
/// checked, at runtime, to be an array of objects.  Returns true if successful,
/// false if unsuccessful.
MLIR_CAPI_EXPORTED bool
firrtlImportAnnotationsFromJSONRaw(MlirContext ctx,
                                   MlirStringRef annotationsStr,
                                   MlirAttribute *importedAnnotationsArray);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_FIRRTL_H
