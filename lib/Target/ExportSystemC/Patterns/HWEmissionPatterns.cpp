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
#include "mlir/IR/BuiltinTypes.h"

using namespace circt;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Type emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit integer types. There are several datatypes to represent integers in
/// SystemC. In contrast to HW and Comb, SystemC chooses the signedness
/// semantics of operations not by the operation itself, but by the type of the
/// operands. We generally map the signless integers of HW to unsigned integers
/// in SystemC and cast to/from signed integers whenever needed. SystemC also
/// uses different types depending on the bit-width of the integer. For 1-bit
/// integers it simply uses 'bool', for integers with up to 64 bits it uses
/// 'sc_uint<>' which maps to native C types for performance reasons. For bigger
/// integers 'sc_biguint<>' is used. However, often a limit of 512 bits is
/// configured for this datatype to improve performance. In that case, we have
/// to fall back to 'sc_bv' bit-vectors which have the disadvantage that many
/// operations such as arithmetics are not supported.
struct IntegerTypeEmitter : TypeEmissionPattern<IntegerType> {
  void emitType(IntegerType type, EmissionPrinter &p) override {
    unsigned bitWidth = type.getIntOrFloatBitWidth();
    if (bitWidth == 1)
      p << "bool";
    else if (bitWidth <= 64)
      p << "sc_dt::sc_uint<" << bitWidth << ">";
    else if (bitWidth <= 512)
      p << "sc_dt::sc_biguint<" << bitWidth << ">";
    else
      p << "sc_dt::sc_bv<" << bitWidth << ">";
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateHWEmitters(OpEmissionPatternSet &patterns,
                                              MLIRContext *context) {}

void circt::ExportSystemC::populateHWTypeEmitters(
    TypeEmissionPatternSet &patterns) {
  patterns.add<IntegerTypeEmitter>();
}
