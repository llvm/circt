//===- Arc.cpp - C interface for the Arc dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Arc.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Arc/ArcTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt::arc;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Arc, arc, circt::arc::ArcDialect)

void registerArcPasses() { circt::arc::registerPasses(); }

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

bool arcTypeIsAState(MlirType type) {
  return llvm::isa<StateType>(unwrap(type));
}

MlirType arcStateTypeGet(MlirType innerType) {
  return wrap(StateType::get(unwrap(innerType)));
}

MlirType arcStateTypeGetType(MlirType type) {
  return wrap(llvm::cast<StateType>(unwrap(type)).getType());
}

unsigned arcStateTypeGetBitWidth(MlirType type) {
  return llvm::cast<StateType>(unwrap(type)).getBitWidth();
}

unsigned arcStateTypeGetByteWidth(MlirType type) {
  return llvm::cast<StateType>(unwrap(type)).getByteWidth();
}

bool arcTypeIsAMemory(MlirType type) {
  return llvm::isa<MemoryType>(unwrap(type));
}

MlirType arcMemoryTypeGet(unsigned numWords, MlirType wordType,
                          MlirType addressType) {
  return wrap(
      MemoryType::get(unwrap(wordType).getContext(), numWords,
                      llvm::cast<mlir::IntegerType>(unwrap(wordType)),
                      llvm::cast<mlir::IntegerType>(unwrap(addressType))));
}

unsigned arcMemoryTypeGetNumWords(MlirType type) {
  return llvm::cast<MemoryType>(unwrap(type)).getNumWords();
}

MlirType arcMemoryTypeGetWordType(MlirType type) {
  return wrap(llvm::cast<MemoryType>(unwrap(type)).getWordType());
}

MlirType arcMemoryTypeGetAddressType(MlirType type) {
  return wrap(llvm::cast<MemoryType>(unwrap(type)).getAddressType());
}

unsigned arcMemoryTypeGetStride(MlirType type) {
  return llvm::cast<MemoryType>(unwrap(type)).getStride();
}

bool arcTypeIsAStorage(MlirType type) {
  return llvm::isa<StorageType>(unwrap(type));
}

MlirType arcStorageTypeGet(MlirContext ctx) {
  return wrap(StorageType::get(unwrap(ctx), 0));
}

MlirType arcStorageTypeGetWithSize(MlirContext ctx, unsigned size) {
  return wrap(StorageType::get(unwrap(ctx), size));
}

unsigned arcStorageTypeGetSize(MlirType type) {
  return llvm::cast<StorageType>(unwrap(type)).getSize();
}

bool arcTypeIsASimModelInstance(MlirType type) {
  return llvm::isa<SimModelInstanceType>(unwrap(type));
}

MlirType arcSimModelInstanceTypeGet(MlirAttribute model) {
  auto attr = llvm::cast<mlir::FlatSymbolRefAttr>(unwrap(model));
  return wrap(SimModelInstanceType::get(attr.getContext(), attr));
}

MlirAttribute arcSimModelInstanceTypeGetModel(MlirType type) {
  return wrap(llvm::cast<SimModelInstanceType>(unwrap(type)).getModel());
}
