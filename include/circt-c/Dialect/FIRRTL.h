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

/// Layer lowering conventions.
// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLLayerConvention {
  FIRRTL_LAYER_CONVENTION_BIND,
  FIRRTL_LAYER_CONVENTION_INLINE,
} FIRRTLLayerConvention;

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

/// Returns `true` if this is a const type whose value is guaranteed to be
/// unchanging at circuit execution time.
MLIR_CAPI_EXPORTED bool firrtlTypeIsConst(MlirType type);

/// Returns a const or non-const version of this type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetConstType(MlirType type, bool isConst);

/// Gets the bit width for this type, returns -1 if unknown.
///
/// It recursively computes the bit width of aggregate types. For bundle and
/// vectors, recursively get the width of each field element and return the
/// total bit width of the aggregate type. This returns -1, if any of the bundle
/// fields is a flip type, or ground type with unknown bit width.
MLIR_CAPI_EXPORTED int64_t firrtlTypeGetBitWidth(MlirType type,
                                                 bool ignoreFlip);

/// Checks if this type is a unsigned integer type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAUInt(MlirType type);

/// Creates a unsigned integer type with the specified width.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetUInt(MlirContext ctx, int32_t width);

/// Checks if this type is a signed integer type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsASInt(MlirType type);

/// Creates a signed integer type with the specified width.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetSInt(MlirContext ctx, int32_t width);

/// Checks if this type is a clock type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAClock(MlirType type);

/// Creates a clock type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetClock(MlirContext ctx);

/// Checks if this type is a reset type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAReset(MlirType type);

/// Creates a reset type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetReset(MlirContext ctx);

/// Checks if this type is an async reset type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAAsyncReset(MlirType type);

/// Creates an async reset type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetAsyncReset(MlirContext ctx);

/// Checks if this type is an analog type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAAnalog(MlirType type);

/// Creates an analog type with the specified width.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetAnalog(MlirContext ctx, int32_t width);

/// Checks if this type is a vector type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAVector(MlirType type);

/// Creates a vector type with the specified element type and count.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetVector(MlirContext ctx,
                                                MlirType element, size_t count);

/// Returns the element type of a vector type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetVectorElement(MlirType vec);

/// Returns the number of elements in a vector type.
MLIR_CAPI_EXPORTED size_t firrtlTypeGetVectorNumElements(MlirType vec);

/// Returns true if the specified type is a bundle type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsABundle(MlirType type);

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

/// Returns the number of fields in the bundle type.
MLIR_CAPI_EXPORTED size_t firrtlTypeGetBundleNumFields(MlirType bundle);

/// Returns the field at the specified index in the bundle type.
MLIR_CAPI_EXPORTED bool
firrtlTypeGetBundleFieldByIndex(MlirType type, size_t index,
                                FIRRTLBundleField *field);

/// Returns the index of the field with the specified name in the bundle type.
MLIR_CAPI_EXPORTED unsigned
firrtlTypeGetBundleFieldIndex(MlirType type, MlirStringRef fieldName);

/// Checks if this type is a ref type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsARef(MlirType type);

/// Creates a ref type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetRef(MlirType target, bool forceable);

/// Creates a ref type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetColoredRef(MlirType target,
                                                    bool forceable,
                                                    MlirAttribute layer);

/// Checks if this type is an anyref type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAAnyRef(MlirType type);

/// Creates an anyref type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetAnyRef(MlirContext ctx);

/// Checks if this type is a property integer type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAInteger(MlirType type);

/// Creates a property integer type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetInteger(MlirContext ctx);

/// Checks if this type is a property double type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsADouble(MlirType type);

/// Creates a property double type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetDouble(MlirContext ctx);

/// Checks if this type is a property string type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAString(MlirType type);

/// Creates a property string type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetString(MlirContext ctx);

/// Checks if this type is a property boolean type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsABoolean(MlirType type);

/// Creates a property boolean type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetBoolean(MlirContext ctx);

/// Checks if this type is a property path type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAPath(MlirType type);

/// Creates a property path type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetPath(MlirContext ctx);

/// Checks if this type is a property list type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAList(MlirType type);

/// Creates a property list type with the specified element type.
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetList(MlirContext ctx,
                                              MlirType elementType);

/// Checks if this type is a class type.
MLIR_CAPI_EXPORTED bool firrtlTypeIsAClass(MlirType type);

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

/// Creates a LayerConventionAttr with the specified value.
MLIR_CAPI_EXPORTED MlirAttribute
firrtlAttrGetLayerConvention(MlirContext ctx, FIRRTLLayerConvention convention);

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

/// Creates a RUWBehaviorAttr with the specified Read-Under-Write Behavior.
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
