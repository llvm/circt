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

MlirType rtgSequenceTypeGet(MlirContext ctxt, intptr_t numArgs,
                            MlirType const *argTypes) {
  SmallVector<Type> types;
  for (unsigned i = 0; i < numArgs; ++i)
    types.push_back(unwrap(argTypes[i]));
  return wrap(SequenceType::get(unwrap(ctxt), types));
}

// ModeType
//===----------------------------------------------------------------------===//

bool rtgTypeIsAMode(MlirType type) { return isa<ModeType>(unwrap(type)); }

MlirType rtgModeTypeGet(MlirContext ctxt) {
  return wrap(ModeType::get(unwrap(ctxt)));
}

// ContextResourceType
//===----------------------------------------------------------------------===//

bool rtgTypeIsAContextResource(MlirType type) {
  return isa<ContextResourceType>(unwrap(type));
}

MlirType rtgContextResourceTypeGet(MlirContext ctxt) {
  return wrap(ContextResourceType::get(unwrap(ctxt)));
}

// SetType
//===----------------------------------------------------------------------===//

bool rtgTypeIsASet(MlirType type) { return isa<SetType>(unwrap(type)); }

MlirType rtgSetTypeGet(MlirContext ctxt, MlirType elementType) {
  return wrap(SetType::get(unwrap(ctxt), unwrap(elementType)));
}

// TargetType
//===----------------------------------------------------------------------===//

bool rtgTypeIsATarget(MlirType type) { return isa<TargetType>(unwrap(type)); }

MlirType rtgTargetTypeGet(MlirContext ctxt, intptr_t numEntries,
                          MlirAttribute const *entryNames,
                          MlirType const *entryTypes) {
  SmallVector<StringAttr> names;
  SmallVector<Type> types;
  for (unsigned i = 0; i < numEntries; ++i) {
    names.push_back(cast<StringAttr>(unwrap(entryNames[i])));
    types.push_back(unwrap(entryTypes[i]));
  }
  return wrap(TargetType::get(unwrap(ctxt), names, types));
}
