//===- ArcTypes.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcTypes.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace arc;
using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Arc/ArcTypes.cpp.inc"

unsigned StateType::getBitWidth() { return hw::getBitWidth(getType()); }

LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (hw::getBitWidth(innerType) < 0)
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}

unsigned MemoryType::getStride() {
  unsigned stride = (getWordType().getWidth() + 7) / 8;
  return llvm::alignToPowerOf2(stride, llvm::bit_ceil(std::min(stride, 16U)));
}

Type MemoryInitializerType::parse(AsmParser &odsParser) {
  unsigned numWords = 0;
  IntegerType wordType;

  if (odsParser.parseLess())
    return {};

  if (odsParser.parseOptionalStar()) {
    auto numLoc = odsParser.getCurrentLocation();
    if (odsParser.parseInteger(numWords))
      return {};
    if (numWords == 0) {
      odsParser.emitError(numLoc, "Number of words must not be zero.");
      return {};
    }
  }

  if (odsParser.parseXInDimensionList() ||
      (odsParser.parseOptionalStar() && odsParser.parseType(wordType)) ||
      odsParser.parseGreater())
    return {};

  return MemoryInitializerType::get(odsParser.getContext(), numWords, wordType);
}

void MemoryInitializerType::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  if (getNumWords() > 0)
    odsPrinter << getNumWords();
  else
    odsPrinter << "*";
  odsPrinter << " x ";
  if (getWordType())
    odsPrinter << getWordType();
  else
    odsPrinter << "*";
  odsPrinter << ">";
}

void ArcDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Arc/ArcTypes.cpp.inc"
      >();
}
