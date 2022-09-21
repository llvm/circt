//===- HWEmissionPatterns.cpp - HW Dialect Emission Patterns --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the HW dialect.
//
//===----------------------------------------------------------------------===//

#include "HWEmissionPatterns.h"
#include "../EmissionPrinter.h"
#include "circt/Dialect/HW/HWOps.h"

using namespace circt;
using namespace circt::hw;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// The ConstantOp always inlines its value. Examples:
/// * hw.constant 5 : i32 ==> 5
/// * hw.constant 0 : i1 ==> false
/// * hw.constant 1 : i1 ==> true
struct ConstantEmitter : OpEmissionPattern<ConstantOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<ConstantOp>())
      return Precedence::LIT;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    APInt val = value.getDefiningOp<ConstantOp>().getValue();

    if (val.getBitWidth() == 1) {
      p << (val.getBoolValue() ? "true" : "false");
      return;
    }

    SmallString<64> valueString;
    val.toStringUnsigned(valueString);
    p << valueString;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Type emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit the builtin integer type to native C integer types.
struct IntegerTypeEmitter : TypeEmissionPattern<IntegerType> {
  bool match(Type type) override {
    if (!type.isa<IntegerType>())
      return false;

    unsigned bw = type.getIntOrFloatBitWidth();
    return bw == 1 || bw == 8 || bw == 16 || bw == 32 || bw == 64;
  }

  void emitType(IntegerType type, EmissionPrinter &p) override {
    unsigned bitWidth = type.getIntOrFloatBitWidth();
    switch (bitWidth) {
    case 1:
      p << "bool";
      break;
    case 8:
    case 16:
    case 32:
    case 64:
      p << (type.isSigned() ? "" : "u") << "int" << bitWidth << "_t";
      break;
    default:
      assert(false && "All cases allowed by match function must be covered.");
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateHWEmitters(OpEmissionPatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<ConstantEmitter>(context);
}

void circt::ExportSystemC::populateHWTypeEmitters(
    TypeEmissionPatternSet &patterns) {
  patterns.add<IntegerTypeEmitter>();
}
