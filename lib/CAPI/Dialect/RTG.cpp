//===- RTG.cpp - C interface for the RTG dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/RTG.h"
#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "circt/Dialect/RTG/IR/RTGTypes.h"
#include "circt/Dialect/RTG/Transforms/RTGPassPipelines.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"

#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"

using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(RTG, rtg, RTGDialect)

void registerRTGPipelines() { circt::rtg::registerPipelines(); }

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

// SequenceType
//===----------------------------------------------------------------------===//

bool rtgTypeIsASequence(MlirType type) {
  return isa<SequenceType>(unwrap(type));
}

MlirType rtgSequenceTypeGet(MlirContext ctxt, intptr_t numElements,
                            MlirType const *elementTypes) {
  SmallVector<Type> types;
  for (unsigned i = 0; i < numElements; ++i)
    types.emplace_back(unwrap(elementTypes[i]));
  return wrap(SequenceType::get(unwrap(ctxt), types));
}

unsigned rtgSequenceTypeGetNumElements(MlirType type) {
  return cast<SequenceType>(unwrap(type)).getElementTypes().size();
}

MlirType rtgSequenceTypeGetElement(MlirType type, unsigned i) {
  return wrap(cast<SequenceType>(unwrap(type)).getElementTypes()[i]);
}

// RandomizedSequenceType
//===----------------------------------------------------------------------===//

bool rtgTypeIsARandomizedSequence(MlirType type) {
  return isa<RandomizedSequenceType>(unwrap(type));
}

MlirType rtgRandomizedSequenceTypeGet(MlirContext ctxt) {
  return wrap(RandomizedSequenceType::get(unwrap(ctxt)));
}

// LabelType
//===----------------------------------------------------------------------===//

bool rtgTypeIsALabel(MlirType type) { return isa<LabelType>(unwrap(type)); }

MlirType rtgLabelTypeGet(MlirContext ctxt) {
  return wrap(LabelType::get(unwrap(ctxt)));
}

// SetType
//===----------------------------------------------------------------------===//

bool rtgTypeIsASet(MlirType type) { return isa<SetType>(unwrap(type)); }

MlirType rtgSetTypeGet(MlirType elementType) {
  auto ty = unwrap(elementType);
  return wrap(SetType::get(ty.getContext(), ty));
}

MlirType rtgSetTypeGetElementType(MlirType type) {
  return wrap(cast<SetType>(unwrap(type)).getElementType());
}

// BagType
//===----------------------------------------------------------------------===//

bool rtgTypeIsABag(MlirType type) { return isa<BagType>(unwrap(type)); }

MlirType rtgBagTypeGet(MlirType elementType) {
  auto ty = unwrap(elementType);
  return wrap(BagType::get(ty.getContext(), ty));
}

MlirType rtgBagTypeGetElementType(MlirType type) {
  return wrap(cast<BagType>(unwrap(type)).getElementType());
}

// DictType
//===----------------------------------------------------------------------===//

bool rtgTypeIsADict(MlirType type) { return isa<DictType>(unwrap(type)); }

MlirType rtgDictTypeGet(MlirContext ctxt, intptr_t numEntries,
                        MlirAttribute const *entryNames,
                        MlirType const *entryTypes) {
  SmallVector<DictEntry> entries;
  for (unsigned i = 0; i < numEntries; ++i) {
    DictEntry entry;
    entry.name = cast<StringAttr>(unwrap(entryNames[i]));
    entry.type = unwrap(entryTypes[i]);
    entries.emplace_back(entry);
  }
  return wrap(DictType::get(unwrap(ctxt), entries));
}

// ArrayType
//===----------------------------------------------------------------------===//

MlirType rtgArrayTypeGet(MlirType elementType) {
  return wrap(
      ArrayType::get(unwrap(elementType).getContext(), unwrap(elementType)));
}

bool rtgTypeIsAArray(MlirType type) { return isa<ArrayType>(unwrap(type)); }

MlirType rtgArrayTypeGetElementType(MlirType type) {
  return wrap(cast<ArrayType>(unwrap(type)).getElementType());
}

// MapType
//===----------------------------------------------------------------------===//

bool rtgTypeIsAMap(MlirType type) { return isa<MapType>(unwrap(type)); }

MlirType rtgMapTypeGet(MlirType keyType, MlirType valueType) {
  return wrap(MapType::get(unwrap(keyType), unwrap(valueType)));
}

MlirType rtgMapTypeGetKeyType(MlirType type) {
  return wrap(cast<MapType>(unwrap(type)).getKeyType());
}

MlirType rtgMapTypeGetValueType(MlirType type) {
  return wrap(cast<MapType>(unwrap(type)).getValueType());
}

// TupleType
//===----------------------------------------------------------------------===//

MlirType rtgTupleTypeGet(MlirContext ctxt, intptr_t numFields,
                         MlirType const *fieldTypes) {
  SmallVector<Type> types;
  for (unsigned i = 0; i < numFields; ++i)
    types.emplace_back(unwrap(fieldTypes[i]));
  return wrap(rtg::TupleType::get(unwrap(ctxt), types));
}

bool rtgTypeIsATuple(MlirType type) {
  return isa<rtg::TupleType>(unwrap(type));
}

intptr_t rtgTypeGetNumFields(MlirType type) {
  return cast<rtg::TupleType>(unwrap(type)).getFieldTypes().size();
}

MlirType rtgTupleTypeGetFieldType(MlirType type, intptr_t idx) {
  return wrap(cast<rtg::TupleType>(unwrap(type)).getFieldTypes()[idx]);
}

// ImmediateType
//===----------------------------------------------------------------------===//

bool rtgTypeIsAImmediate(MlirType type) {
  return isa<ImmediateType>(unwrap(type));
}

MlirType rtgImmediateTypeGet(MlirContext ctx, uint32_t width) {
  return wrap(ImmediateType::get(unwrap(ctx), width));
}

uint32_t rtgImmediateTypeGetWidth(MlirType type) {
  return cast<ImmediateType>(unwrap(type)).getWidth();
}

// MemoryType
//===----------------------------------------------------------------------===//

bool rtgTypeIsAMemory(MlirType type) { return isa<MemoryType>(unwrap(type)); }

MlirType rtgMemoryTypeGet(MlirContext ctxt, uint32_t addressWidth) {
  return wrap(MemoryType::get(unwrap(ctxt), addressWidth));
}

uint32_t rtgMemoryTypeGetAddressWidth(MlirType type) {
  return cast<MemoryType>(unwrap(type)).getAddressWidth();
}

// MemoryBlockType
//===----------------------------------------------------------------------===//

bool rtgTypeIsAMemoryBlock(MlirType type) {
  return isa<MemoryBlockType>(unwrap(type));
}

MlirType rtgMemoryBlockTypeGet(MlirContext ctxt, uint32_t addressWidth) {
  return wrap(MemoryBlockType::get(unwrap(ctxt), addressWidth));
}

uint32_t rtgMemoryBlockTypeGetAddressWidth(MlirType type) {
  return cast<MemoryBlockType>(unwrap(type)).getAddressWidth();
}

// StringType
//===----------------------------------------------------------------------===//

bool rtgTypeIsAString(MlirType type) { return isa<StringType>(unwrap(type)); }

MlirType rtgStringTypeGet(MlirContext ctxt) {
  return wrap(StringType::get(unwrap(ctxt)));
}

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

// DefaultContext
//===----------------------------------------------------------------------===//

bool rtgAttrIsADefaultContextAttr(MlirAttribute attr) {
  return isa<DefaultContextAttr>(unwrap(attr));
}

MlirAttribute rtgDefaultContextAttrGet(MlirContext ctxt, MlirType type) {
  return wrap(DefaultContextAttr::get(unwrap(ctxt), unwrap(type)));
}

// Label Visibility
//===----------------------------------------------------------------------===//

bool rtgAttrIsALabelVisibilityAttr(MlirAttribute attr) {
  return isa<LabelVisibilityAttr>(unwrap(attr));
}

RTGLabelVisibility rtgLabelVisibilityAttrGetValue(MlirAttribute attr) {
  auto convert = [](LabelVisibility visibility) {
    switch (visibility) {
    case LabelVisibility::local:
      return RTG_LABEL_VISIBILITY_LOCAL;
    case LabelVisibility::global:
      return RTG_LABEL_VISIBILITY_GLOBAL;
    case LabelVisibility::external:
      return RTG_LABEL_VISIBILITY_EXTERNAL;
    }
  };
  return convert(cast<LabelVisibilityAttr>(unwrap(attr)).getValue());
}

MlirAttribute rtgLabelVisibilityAttrGet(MlirContext ctxt,
                                        RTGLabelVisibility visibility) {
  auto convert = [](RTGLabelVisibility visibility) {
    switch (visibility) {
    case RTG_LABEL_VISIBILITY_LOCAL:
      return LabelVisibility::local;
    case RTG_LABEL_VISIBILITY_GLOBAL:
      return LabelVisibility::global;
    case RTG_LABEL_VISIBILITY_EXTERNAL:
      return LabelVisibility::external;
    }
  };
  return wrap(LabelVisibilityAttr::get(unwrap(ctxt), convert(visibility)));
}

// ImmediateAttr
//===----------------------------------------------------------------------===//

bool rtgAttrIsAImmediate(MlirAttribute attr) {
  return isa<ImmediateAttr>(unwrap(attr));
}

MlirAttribute rtgImmediateAttrGet(MlirContext ctx, uint32_t width,
                                  uint64_t value) {
  return wrap(rtg::ImmediateAttr::get(unwrap(ctx), APInt(width, value)));
}

uint32_t rtgImmediateAttrGetWidth(MlirAttribute attr) {
  return cast<ImmediateAttr>(unwrap(attr)).getValue().getBitWidth();
}

uint64_t rtgImmediateAttrGetValue(MlirAttribute attr) {
  return cast<ImmediateAttr>(unwrap(attr)).getValue().getZExtValue();
}

// AnyContexts
//===----------------------------------------------------------------------===//

bool rtgAttrIsAAnyContextAttr(MlirAttribute attr) {
  return isa<AnyContextAttr>(unwrap(attr));
}

MlirAttribute rtgAnyContextAttrGet(MlirContext ctxt, MlirType type) {
  return wrap(AnyContextAttr::get(unwrap(ctxt), unwrap(type)));
}

// VirtualRegConfigAttr
//===----------------------------------------------------------------------===//

bool rtgAttrIsAVirtualRegisterConfig(MlirAttribute attr) {
  return isa<VirtualRegisterConfigAttr>(unwrap(attr));
}

MlirAttribute
rtgVirtualRegisterConfigAttrGet(MlirContext ctxt, intptr_t numRegs,
                                MlirAttribute const *allowedRegs) {
  SmallVector<rtg::RegisterAttrInterface> regs;
  for (intptr_t i = 0; i < numRegs; ++i)
    regs.push_back(cast<rtg::RegisterAttrInterface>(unwrap(allowedRegs[i])));
  return wrap(VirtualRegisterConfigAttr::get(unwrap(ctxt), regs));
}

intptr_t rtgVirtualRegisterConfigAttrGetNumRegisters(MlirAttribute attr) {
  return cast<VirtualRegisterConfigAttr>(unwrap(attr)).getAllowedRegs().size();
}

MlirAttribute rtgVirtualRegisterConfigAttrGetRegister(MlirAttribute attr,
                                                      intptr_t index) {
  auto allowedRegs =
      cast<VirtualRegisterConfigAttr>(unwrap(attr)).getAllowedRegs();
  return wrap(allowedRegs[index]);
}

// LabelAttr
//===----------------------------------------------------------------------===//

bool rtgAttrIsALabel(MlirAttribute attr) {
  return isa<LabelAttr>(unwrap(attr));
}

MlirAttribute rtgLabelAttrGet(MlirContext ctx, MlirStringRef name) {
  return wrap(LabelAttr::get(unwrap(ctx), unwrap(name)));
}

MlirStringRef rtgLabelAttrGetName(MlirAttribute attr) {
  return wrap(cast<LabelAttr>(unwrap(attr)).getName());
}

// MapAttr
//===----------------------------------------------------------------------===//

bool rtgAttrIsAMap(MlirAttribute attr) {
  return isa<rtg::MapAttr>(unwrap(attr));
}

MlirAttribute rtgMapAttrGet(MlirContext ctx, MlirType mapType,
                            intptr_t numEntries, MlirAttribute const *keys,
                            MlirAttribute const *values) {
  DenseMap<TypedAttr, TypedAttr> entries;
  for (unsigned i = 0; i < numEntries; ++i) {
    entries.insert(
        {cast<TypedAttr>(unwrap(keys[i])), cast<TypedAttr>(unwrap(values[i]))});
  }
  return wrap(rtg::MapAttr::get(cast<rtg::MapType>(unwrap(mapType)), &entries));
}

MlirAttribute rtgMapAttrLookup(MlirAttribute attr, MlirAttribute key) {
  auto mapAttr = cast<rtg::MapAttr>(unwrap(attr));
  auto keyAttr = cast<TypedAttr>(unwrap(key));

  auto it = mapAttr.getEntries()->find(keyAttr);
  if (it == mapAttr.getEntries()->end())
    return wrap(Attribute()); // Return null attribute if key not found

  return wrap(it->second);
}

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/Transforms/RTGPasses.capi.cpp.inc"
