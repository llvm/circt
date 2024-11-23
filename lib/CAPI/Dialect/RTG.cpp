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

#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/CAPI/Registration.h"

using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(RTG, rtg, RTGDialect)
void registerRTGPasses() { registerPasses(); }

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

// SetType
//===----------------------------------------------------------------------===//

bool rtgTypeIsASet(MlirType type) { return isa<SetType>(unwrap(type)); }

MlirType rtgSetTypeGet(MlirType elementType) {
  auto ty = unwrap(elementType);
  return wrap(SetType::get(ty.getContext(), ty));
}

// DictType
//===----------------------------------------------------------------------===//

bool rtgTypeIsADict(MlirType type) { return isa<DictType>(unwrap(type)); }

MlirType rtgDictTypeGet(MlirContext ctxt, intptr_t numEntries,
                        MlirAttribute const *entryNames,
                        MlirType const *entryTypes) {
  SmallVector<StringAttr> names;
  SmallVector<Type> types;
  for (unsigned i = 0; i < numEntries; ++i) {
    names.push_back(cast<StringAttr>(unwrap(entryNames[i])));
    types.push_back(unwrap(entryTypes[i]));
  }
  return wrap(DictType::get(unwrap(ctxt), names, types));
}
