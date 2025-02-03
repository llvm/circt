//===- RTG.cpp - C interface for the RTG dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/RTG.h"
#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "circt/Dialect/RTG/IR/RTGTypes.h"

#include "mlir/CAPI/Registration.h"

using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(RTG, rtg, RTGDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

// SequenceType
//===----------------------------------------------------------------------===//

bool rtgTypeIsASequence(MlirType type) {
  return isa<SequenceType>(unwrap(type));
}

MlirType rtgSequenceTypeGet(MlirContext ctxt) {
  return wrap(SequenceType::get(unwrap(ctxt)));
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

// BagType
//===----------------------------------------------------------------------===//

bool rtgTypeIsABag(MlirType type) { return isa<BagType>(unwrap(type)); }

MlirType rtgBagTypeGet(MlirType elementType) {
  auto ty = unwrap(elementType);
  return wrap(BagType::get(ty.getContext(), ty));
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

//===----------------------------------------------------------------------===//
// Attribute API.
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
