//===- HW.h - C interface for the HW dialect ----------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_HW_H
#define CIRCT_C_DIALECT_HW_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(HWInstanceGraph, void);
DEFINE_C_API_STRUCT(HWInstanceGraphNode, void);

#undef DEFINE_C_API_STRUCT

struct HWStructFieldInfo {
  MlirIdentifier name;
  MlirType type;
};
typedef struct HWStructFieldInfo HWStructFieldInfo;

struct HWUnionFieldInfo {
  MlirIdentifier name;
  MlirType type;
  size_t offset;
};
typedef struct HWUnionFieldInfo HWUnionFieldInfo;

enum HWModulePortDirection { Input, Output, InOut };
typedef enum HWModulePortDirection HWModulePortDirection;

struct HWModulePort {
  MlirAttribute name;
  MlirType type;
  HWModulePortDirection dir;
};
typedef struct HWModulePort HWModulePort;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HW, hw);
MLIR_CAPI_EXPORTED void registerHWPasses(void);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Return the hardware bit width of a type. Does not reflect any encoding,
/// padding, or storage scheme, just the bit (and wire width) of a
/// statically-size type. Reflects the number of wires needed to transmit a
/// value of this type. Returns -1 if the type is not known or cannot be
/// statically computed.
MLIR_CAPI_EXPORTED int64_t hwGetBitWidth(MlirType);

/// Return true if the specified type can be used as an HW value type, that is
/// the set of types that can be composed together to represent synthesized,
/// hardware but not marker types like InOutType or unknown types from other
/// dialects.
MLIR_CAPI_EXPORTED bool hwTypeIsAValueType(MlirType);

/// If the type is an HW array
MLIR_CAPI_EXPORTED bool hwTypeIsAArrayType(MlirType);

/// If the type is an HW inout.
MLIR_CAPI_EXPORTED bool hwTypeIsAInOut(MlirType type);

/// If the type is an HW module type.
MLIR_CAPI_EXPORTED bool hwTypeIsAModuleType(MlirType type);

/// If the type is an HW struct.
MLIR_CAPI_EXPORTED bool hwTypeIsAStructType(MlirType);

/// If the type is an HW union.
MLIR_CAPI_EXPORTED bool hwTypeIsAUnionType(MlirType);

/// If the type is an HW type alias.
MLIR_CAPI_EXPORTED bool hwTypeIsATypeAliasType(MlirType);

/// If the type is an HW int.
MLIR_CAPI_EXPORTED bool hwTypeIsAIntType(MlirType);

/// Creates a fixed-size HW array type in the context associated with element
MLIR_CAPI_EXPORTED MlirType hwArrayTypeGet(MlirType element, size_t size);

/// returns the element type of an array type
MLIR_CAPI_EXPORTED MlirType hwArrayTypeGetElementType(MlirType);

/// returns the size of an array type
MLIR_CAPI_EXPORTED intptr_t hwArrayTypeGetSize(MlirType);

/// Creates an HW inout type in the context associated with element.
MLIR_CAPI_EXPORTED MlirType hwInOutTypeGet(MlirType element);

/// Returns the element type of an inout type.
MLIR_CAPI_EXPORTED MlirType hwInOutTypeGetElementType(MlirType);

/// Creates an HW module type.
MLIR_CAPI_EXPORTED MlirType hwModuleTypeGet(MlirContext ctx, intptr_t numPorts,
                                            HWModulePort const *ports);

/// Get an HW module type's number of inputs.
MLIR_CAPI_EXPORTED intptr_t hwModuleTypeGetNumInputs(MlirType type);

/// Get an HW module type's input type at a specific index.
MLIR_CAPI_EXPORTED MlirType hwModuleTypeGetInputType(MlirType type,
                                                     intptr_t index);

/// Get an HW module type's input name at a specific index.
MLIR_CAPI_EXPORTED MlirStringRef hwModuleTypeGetInputName(MlirType type,
                                                          intptr_t index);

/// Get an HW module type's number of outputs.
MLIR_CAPI_EXPORTED intptr_t hwModuleTypeGetNumOutputs(MlirType type);

/// Get an HW module type's output type at a specific index.
MLIR_CAPI_EXPORTED MlirType hwModuleTypeGetOutputType(MlirType type,
                                                      intptr_t index);

/// Get an HW module type's output name at a specific index.
MLIR_CAPI_EXPORTED MlirStringRef hwModuleTypeGetOutputName(MlirType type,
                                                           intptr_t index);

/// Get an HW module type's port info at a specific index.
MLIR_CAPI_EXPORTED void hwModuleTypeGetPort(MlirType type, intptr_t index,
                                            HWModulePort *ret);

/// Creates an HW struct type in the context associated with the elements.
MLIR_CAPI_EXPORTED MlirType hwStructTypeGet(MlirContext ctx,
                                            intptr_t numElements,
                                            HWStructFieldInfo const *elements);

MLIR_CAPI_EXPORTED MlirType hwStructTypeGetField(MlirType structType,
                                                 MlirStringRef fieldName);

MLIR_CAPI_EXPORTED MlirType hwParamIntTypeGet(MlirAttribute parameter);

MLIR_CAPI_EXPORTED MlirAttribute hwParamIntTypeGetWidthAttr(MlirType);

MLIR_CAPI_EXPORTED MlirAttribute
hwStructTypeGetFieldIndex(MlirType structType, MlirStringRef fieldName);

MLIR_CAPI_EXPORTED HWStructFieldInfo
hwStructTypeGetFieldNum(MlirType structType, unsigned idx);

MLIR_CAPI_EXPORTED intptr_t hwStructTypeGetNumFields(MlirType structType);

/// Creates an HW union type in the context associated with the elements.
MLIR_CAPI_EXPORTED MlirType hwUnionTypeGet(MlirContext ctx,
                                           intptr_t numElements,
                                           HWUnionFieldInfo const *elements);

MLIR_CAPI_EXPORTED MlirType hwUnionTypeGetField(MlirType unionType,
                                                MlirStringRef fieldName);

MLIR_CAPI_EXPORTED MlirAttribute
hwUnionTypeGetFieldIndex(MlirType unionType, MlirStringRef fieldName);

MLIR_CAPI_EXPORTED HWUnionFieldInfo hwUnionTypeGetFieldNum(MlirType unionType,
                                                           unsigned idx);

MLIR_CAPI_EXPORTED intptr_t hwUnionTypeGetNumFields(MlirType unionType);

MLIR_CAPI_EXPORTED MlirType hwTypeAliasTypeGet(MlirStringRef scope,
                                               MlirStringRef name,
                                               MlirType innerType);

MLIR_CAPI_EXPORTED MlirType hwTypeAliasTypeGetCanonicalType(MlirType typeAlias);

MLIR_CAPI_EXPORTED MlirType hwTypeAliasTypeGetInnerType(MlirType typeAlias);

MLIR_CAPI_EXPORTED MlirStringRef hwTypeAliasTypeGetName(MlirType typeAlias);

MLIR_CAPI_EXPORTED MlirStringRef hwTypeAliasTypeGetScope(MlirType typeAlias);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool hwAttrIsAInnerSymAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwInnerSymAttrGet(MlirAttribute symName);
MLIR_CAPI_EXPORTED MlirAttribute hwInnerSymAttrGetEmpty(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirAttribute hwInnerSymAttrGetSymName(MlirAttribute);

MLIR_CAPI_EXPORTED bool hwAttrIsAInnerRefAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwInnerRefAttrGet(MlirAttribute moduleName,
                                                   MlirAttribute innerSym);
MLIR_CAPI_EXPORTED MlirAttribute hwInnerRefAttrGetName(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwInnerRefAttrGetModule(MlirAttribute);

MLIR_CAPI_EXPORTED bool hwAttrIsAParamDeclAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwParamDeclAttrGet(MlirStringRef name,
                                                    MlirType type,
                                                    MlirAttribute value);
MLIR_CAPI_EXPORTED MlirStringRef hwParamDeclAttrGetName(MlirAttribute decl);
MLIR_CAPI_EXPORTED MlirType hwParamDeclAttrGetType(MlirAttribute decl);
MLIR_CAPI_EXPORTED MlirAttribute hwParamDeclAttrGetValue(MlirAttribute decl);

MLIR_CAPI_EXPORTED bool hwAttrIsAParamDeclRefAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwParamDeclRefAttrGet(MlirContext ctx,
                                                       MlirStringRef cName);
MLIR_CAPI_EXPORTED MlirStringRef hwParamDeclRefAttrGetName(MlirAttribute decl);
MLIR_CAPI_EXPORTED MlirType hwParamDeclRefAttrGetType(MlirAttribute decl);

MLIR_CAPI_EXPORTED bool hwAttrIsAParamVerbatimAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwParamVerbatimAttrGet(MlirAttribute text);

MLIR_CAPI_EXPORTED bool hwAttrIsAOutputFileAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwOutputFileGetFromFileName(
    MlirAttribute text, bool excludeFromFileList, bool includeReplicatedOp);
MLIR_CAPI_EXPORTED MlirStringRef
hwOutputFileGetFileName(MlirAttribute outputFile);

//===----------------------------------------------------------------------===//
// InstanceGraph API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED HWInstanceGraph hwInstanceGraphGet(MlirOperation operation);

MLIR_CAPI_EXPORTED void hwInstanceGraphDestroy(HWInstanceGraph instanceGraph);

MLIR_CAPI_EXPORTED HWInstanceGraphNode
hwInstanceGraphGetTopLevelNode(HWInstanceGraph instanceGraph);

// NOLINTNEXTLINE(modernize-use-using)
typedef void (*HWInstanceGraphNodeCallback)(HWInstanceGraphNode, void *);

MLIR_CAPI_EXPORTED void
hwInstanceGraphForEachNode(HWInstanceGraph instanceGraph,
                           HWInstanceGraphNodeCallback callback,
                           void *userData);

MLIR_CAPI_EXPORTED bool hwInstanceGraphNodeEqual(HWInstanceGraphNode lhs,
                                                 HWInstanceGraphNode rhs);

MLIR_CAPI_EXPORTED MlirModule
hwInstanceGraphNodeGetModule(HWInstanceGraphNode node);

MLIR_CAPI_EXPORTED MlirOperation
hwInstanceGraphNodeGetModuleOp(HWInstanceGraphNode node);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_HW_H
