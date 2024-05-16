//===- HW.cpp - C interface for the HW dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/HW.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "llvm/ADT/PostOrderIterator.h"

using namespace circt;
using namespace circt::hw;

DEFINE_C_API_PTR_METHODS(HWInstanceGraph, InstanceGraph)
DEFINE_C_API_PTR_METHODS(HWInstanceGraphNode, igraph::InstanceGraphNode)

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HW, hw, HWDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

int64_t hwGetBitWidth(MlirType type) { return getBitWidth(unwrap(type)); }

bool hwTypeIsAValueType(MlirType type) { return isHWValueType(unwrap(type)); }

bool hwTypeIsAArrayType(MlirType type) { return isa<ArrayType>(unwrap(type)); }

MlirType hwArrayTypeGet(MlirType element, size_t size) {
  return wrap(ArrayType::get(unwrap(element), size));
}

MlirType hwArrayTypeGetElementType(MlirType type) {
  return wrap(cast<ArrayType>(unwrap(type)).getElementType());
}

intptr_t hwArrayTypeGetSize(MlirType type) {
  return cast<ArrayType>(unwrap(type)).getNumElements();
}

bool hwTypeIsAIntType(MlirType type) { return isa<IntType>(unwrap(type)); }

MlirType hwParamIntTypeGet(MlirAttribute parameter) {
  return wrap(IntType::get(cast<TypedAttr>(unwrap(parameter))));
}

MlirAttribute hwParamIntTypeGetWidthAttr(MlirType type) {
  return wrap(cast<IntType>(unwrap(type)).getWidth());
}

MlirType hwInOutTypeGet(MlirType element) {
  return wrap(InOutType::get(unwrap(element)));
}

MlirType hwInOutTypeGetElementType(MlirType type) {
  return wrap(cast<InOutType>(unwrap(type)).getElementType());
}

bool hwTypeIsAInOut(MlirType type) { return isa<InOutType>(unwrap(type)); }

bool hwTypeIsAModuleType(MlirType type) {
  return isa<ModuleType>(unwrap(type));
}

MlirType hwModuleTypeGet(MlirContext ctx, intptr_t numPorts,
                         HWModulePort const *ports) {
  SmallVector<ModulePort> modulePorts;
  for (intptr_t i = 0; i < numPorts; ++i) {
    HWModulePort port = ports[i];

    ModulePort::Direction dir;
    switch (port.dir) {
    case HWModulePortDirection::Input:
      dir = ModulePort::Direction::Input;
      break;
    case HWModulePortDirection::Output:
      dir = ModulePort::Direction::Output;
      break;
    case HWModulePortDirection::InOut:
      dir = ModulePort::Direction::InOut;
      break;
    }

    StringAttr name = cast<StringAttr>(unwrap(port.name));
    Type type = unwrap(port.type);

    modulePorts.push_back(ModulePort{name, type, dir});
  }

  return wrap(ModuleType::get(unwrap(ctx), modulePorts));
}

intptr_t hwModuleTypeGetNumInputs(MlirType type) {
  return cast<ModuleType>(unwrap(type)).getNumInputs();
}

MlirType hwModuleTypeGetInputType(MlirType type, intptr_t index) {
  return wrap(cast<ModuleType>(unwrap(type)).getInputType(index));
}

MlirStringRef hwModuleTypeGetInputName(MlirType type, intptr_t index) {
  return wrap(cast<ModuleType>(unwrap(type)).getInputName(index));
}

intptr_t hwModuleTypeGetNumOutputs(MlirType type) {
  return cast<ModuleType>(unwrap(type)).getNumOutputs();
}

MlirType hwModuleTypeGetOutputType(MlirType type, intptr_t index) {
  return wrap(cast<ModuleType>(unwrap(type)).getOutputType(index));
}

MlirStringRef hwModuleTypeGetOutputName(MlirType type, intptr_t index) {
  return wrap(cast<ModuleType>(unwrap(type)).getOutputName(index));
}

bool hwTypeIsAStructType(MlirType type) {
  return isa<StructType>(unwrap(type));
}

MlirType hwStructTypeGet(MlirContext ctx, intptr_t numElements,
                         HWStructFieldInfo const *elements) {
  SmallVector<StructType::FieldInfo> fieldInfos;
  fieldInfos.reserve(numElements);
  for (intptr_t i = 0; i < numElements; ++i) {
    fieldInfos.push_back(StructType::FieldInfo{
        cast<StringAttr>(unwrap(elements[i].name)), unwrap(elements[i].type)});
  }
  return wrap(StructType::get(unwrap(ctx), fieldInfos));
}

MlirType hwStructTypeGetField(MlirType structType, MlirStringRef fieldName) {
  StructType st = cast<StructType>(unwrap(structType));
  return wrap(st.getFieldType(unwrap(fieldName)));
}

MlirAttribute hwStructTypeGetFieldIndex(MlirType structType,
                                        MlirStringRef fieldName) {
  StructType st = cast<StructType>(unwrap(structType));
  if (auto idx = st.getFieldIndex(unwrap(fieldName)))
    return wrap(IntegerAttr::get(IntegerType::get(st.getContext(), 32), *idx));
  return wrap(UnitAttr::get(st.getContext()));
}

intptr_t hwStructTypeGetNumFields(MlirType structType) {
  StructType st = cast<StructType>(unwrap(structType));
  return st.getElements().size();
}

HWStructFieldInfo hwStructTypeGetFieldNum(MlirType structType, unsigned idx) {
  StructType st = cast<StructType>(unwrap(structType));
  auto cppField = st.getElements()[idx];
  HWStructFieldInfo ret;
  ret.name = wrap(cppField.name);
  ret.type = wrap(cppField.type);
  return ret;
}

bool hwTypeIsATypeAliasType(MlirType type) {
  return isa<TypeAliasType>(unwrap(type));
}

MlirType hwTypeAliasTypeGet(MlirStringRef cScope, MlirStringRef cName,
                            MlirType cInnerType) {
  StringRef scope = unwrap(cScope);
  StringRef name = unwrap(cName);
  Type innerType = unwrap(cInnerType);
  FlatSymbolRefAttr nameRef =
      FlatSymbolRefAttr::get(innerType.getContext(), name);
  SymbolRefAttr ref =
      SymbolRefAttr::get(innerType.getContext(), scope, {nameRef});
  return wrap(TypeAliasType::get(ref, innerType));
}

MlirType hwTypeAliasTypeGetCanonicalType(MlirType typeAlias) {
  TypeAliasType type = cast<TypeAliasType>(unwrap(typeAlias));
  return wrap(type.getCanonicalType());
}

MlirType hwTypeAliasTypeGetInnerType(MlirType typeAlias) {
  TypeAliasType type = cast<TypeAliasType>(unwrap(typeAlias));
  return wrap(type.getInnerType());
}

MlirStringRef hwTypeAliasTypeGetName(MlirType typeAlias) {
  TypeAliasType type = cast<TypeAliasType>(unwrap(typeAlias));
  return wrap(type.getRef().getLeafReference().getValue());
}

MlirStringRef hwTypeAliasTypeGetScope(MlirType typeAlias) {
  TypeAliasType type = cast<TypeAliasType>(unwrap(typeAlias));
  return wrap(type.getRef().getRootReference().getValue());
}

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

bool hwAttrIsAInnerSymAttr(MlirAttribute attr) {
  return isa<InnerSymAttr>(unwrap(attr));
}

MlirAttribute hwInnerSymAttrGet(MlirAttribute symName) {
  return wrap(InnerSymAttr::get(cast<StringAttr>(unwrap(symName))));
}

MlirAttribute hwInnerSymAttrGetSymName(MlirAttribute innerSymAttr) {
  return wrap((Attribute)cast<InnerSymAttr>(unwrap(innerSymAttr)).getSymName());
}

bool hwAttrIsAInnerRefAttr(MlirAttribute attr) {
  return isa<InnerRefAttr>(unwrap(attr));
}

MlirAttribute hwInnerRefAttrGet(MlirAttribute moduleName,
                                MlirAttribute innerSym) {
  auto moduleNameAttr = cast<StringAttr>(unwrap(moduleName));
  auto innerSymAttr = cast<StringAttr>(unwrap(innerSym));
  return wrap(InnerRefAttr::get(moduleNameAttr, innerSymAttr));
}

MlirAttribute hwInnerRefAttrGetName(MlirAttribute innerRefAttr) {
  return wrap((Attribute)cast<InnerRefAttr>(unwrap(innerRefAttr)).getTarget());
}

MlirAttribute hwInnerRefAttrGetModule(MlirAttribute innerRefAttr) {
  return wrap((Attribute)cast<InnerRefAttr>(unwrap(innerRefAttr)).getRoot());
}

MLIR_CAPI_EXPORTED bool hwAttrIsAParamDeclAttr(MlirAttribute attr) {
  return isa<ParamDeclAttr>(unwrap(attr));
}
MLIR_CAPI_EXPORTED MlirAttribute hwParamDeclAttrGet(MlirStringRef cName,
                                                    MlirType cType,
                                                    MlirAttribute cValue) {
  auto type = unwrap(cType);
  auto name = StringAttr::get(type.getContext(), unwrap(cName));
  return wrap(
      ParamDeclAttr::get(type.getContext(), name, type, unwrap(cValue)));
}
MLIR_CAPI_EXPORTED MlirStringRef hwParamDeclAttrGetName(MlirAttribute decl) {
  return wrap(cast<ParamDeclAttr>(unwrap(decl)).getName().getValue());
}
MLIR_CAPI_EXPORTED MlirType hwParamDeclAttrGetType(MlirAttribute decl) {
  return wrap(cast<ParamDeclAttr>(unwrap(decl)).getType());
}
MLIR_CAPI_EXPORTED MlirAttribute hwParamDeclAttrGetValue(MlirAttribute decl) {
  return wrap(cast<ParamDeclAttr>(unwrap(decl)).getValue());
}

MLIR_CAPI_EXPORTED bool hwAttrIsAParamDeclRefAttr(MlirAttribute attr) {
  return isa<ParamDeclRefAttr>(unwrap(attr));
}

MLIR_CAPI_EXPORTED MlirAttribute hwParamDeclRefAttrGet(MlirContext ctx,
                                                       MlirStringRef cName) {
  auto name = StringAttr::get(unwrap(ctx), unwrap(cName));
  return wrap(ParamDeclRefAttr::get(unwrap(ctx), name,
                                    IntegerType::get(unwrap(ctx), 32)));
}

MLIR_CAPI_EXPORTED MlirStringRef hwParamDeclRefAttrGetName(MlirAttribute decl) {
  return wrap(cast<ParamDeclRefAttr>(unwrap(decl)).getName().getValue());
}
MLIR_CAPI_EXPORTED MlirType hwParamDeclRefAttrGetType(MlirAttribute decl) {
  return wrap(cast<ParamDeclRefAttr>(unwrap(decl)).getType());
}

MLIR_CAPI_EXPORTED bool hwAttrIsAParamVerbatimAttr(MlirAttribute attr) {
  return isa<ParamVerbatimAttr>(unwrap(attr));
}
MLIR_CAPI_EXPORTED MlirAttribute hwParamVerbatimAttrGet(MlirAttribute text) {
  auto textAttr = cast<StringAttr>(unwrap(text));
  MLIRContext *ctx = textAttr.getContext();
  auto type = NoneType::get(ctx);
  return wrap(ParamVerbatimAttr::get(ctx, textAttr, type));
}

MLIR_CAPI_EXPORTED bool hwAttrIsAOutputFileAttr(MlirAttribute attr) {
  return isa<OutputFileAttr>(unwrap(attr));
}
MLIR_CAPI_EXPORTED MlirAttribute
hwOutputFileGetFromFileName(MlirAttribute fileName, bool excludeFromFileList,
                            bool includeReplicatedOp) {
  auto fileNameStrAttr = cast<StringAttr>(unwrap(fileName));
  return wrap(OutputFileAttr::getFromFilename(
      fileNameStrAttr.getContext(), fileNameStrAttr.getValue(),
      excludeFromFileList, includeReplicatedOp));
}

MLIR_CAPI_EXPORTED HWInstanceGraph hwInstanceGraphGet(MlirOperation operation) {
  return wrap(new InstanceGraph{unwrap(operation)});
}

MLIR_CAPI_EXPORTED void hwInstanceGraphDestroy(HWInstanceGraph instanceGraph) {
  delete unwrap(instanceGraph);
}

MLIR_CAPI_EXPORTED HWInstanceGraphNode
hwInstanceGraphGetTopLevelNode(HWInstanceGraph instanceGraph) {
  return wrap(unwrap(instanceGraph)->getTopLevelNode());
}

MLIR_CAPI_EXPORTED void
hwInstanceGraphForEachNode(HWInstanceGraph instanceGraph,
                           HWInstanceGraphNodeCallback callback,
                           void *userData) {
  InstanceGraph *graph = unwrap(instanceGraph);
  for (const auto &inst : llvm::post_order(graph)) {
    callback(wrap(inst), userData);
  }
}

MLIR_CAPI_EXPORTED bool hwInstanceGraphNodeEqual(HWInstanceGraphNode lhs,
                                                 HWInstanceGraphNode rhs) {
  return unwrap(lhs) == unwrap(rhs);
}

MLIR_CAPI_EXPORTED MlirModule
hwInstanceGraphNodeGetModule(HWInstanceGraphNode node) {
  return wrap(dyn_cast<ModuleOp>(unwrap(node)->getModule()));
}

MLIR_CAPI_EXPORTED MlirOperation
hwInstanceGraphNodeGetModuleOp(HWInstanceGraphNode node) {
  return wrap(unwrap(node)->getModule());
}
