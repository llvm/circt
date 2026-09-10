//===- Seq.cpp - C interface for the Seq dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Seq.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Seq/SeqTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt::seq;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Sequential, seq, circt::seq::SeqDialect)

void registerSeqPasses() { circt::seq::registerPasses(); }

bool seqTypeIsAClock(MlirType type) {
  return llvm::isa<ClockType>(unwrap(type));
}

MlirType seqClockTypeGet(MlirContext ctx) {
  return wrap(ClockType::get(unwrap(ctx)));
}

bool seqTypeIsAImmutable(MlirType type) {
  return llvm::isa<ImmutableType>(unwrap(type));
}

MlirType seqImmutableTypeGet(MlirType innerType) {
  return wrap(ImmutableType::get(unwrap(innerType)));
}

MlirType seqImmutableTypeGetInnerType(MlirType type) {
  return wrap(llvm::cast<ImmutableType>(unwrap(type)).getInnerType());
}

bool seqTypeIsAHLMem(MlirType type) {
  return llvm::isa<HLMemType>(unwrap(type));
}

MlirType seqHLMemTypeGet(MlirContext ctx, intptr_t rank, const int64_t *shape,
                         MlirType elementType) {
  llvm::SmallVector<int64_t> shapeVec(shape, shape + rank);
  return wrap(HLMemType::get(unwrap(ctx), shapeVec, unwrap(elementType)));
}

MlirType seqHLMemTypeGetElementType(MlirType type) {
  return wrap(llvm::cast<HLMemType>(unwrap(type)).getElementType());
}

intptr_t seqHLMemTypeGetRank(MlirType type) {
  return llvm::cast<HLMemType>(unwrap(type)).getRank();
}

const int64_t *seqHLMemTypeGetShape(MlirType type) {
  return llvm::cast<HLMemType>(unwrap(type)).getShape().data();
}

bool seqTypeIsAFirMem(MlirType type) {
  return llvm::isa<FirMemType>(unwrap(type));
}

MlirType seqFirMemTypeGet(MlirContext ctx, uint64_t depth, uint32_t width,
                          const uint32_t *maskWidth) {
  std::optional<uint32_t> maskOpt;
  if (maskWidth)
    maskOpt = *maskWidth;
  return wrap(FirMemType::get(unwrap(ctx), depth, width, maskOpt));
}

uint64_t seqFirMemTypeGetDepth(MlirType type) {
  return llvm::cast<FirMemType>(unwrap(type)).getDepth();
}

uint32_t seqFirMemTypeGetWidth(MlirType type) {
  return llvm::cast<FirMemType>(unwrap(type)).getWidth();
}

bool seqFirMemTypeHasMask(MlirType type) {
  return llvm::cast<FirMemType>(unwrap(type)).getMaskWidth().has_value();
}

uint32_t seqFirMemTypeGetMaskWidth(MlirType type) {
  return llvm::cast<FirMemType>(unwrap(type)).getMaskWidth().value_or(0);
}
