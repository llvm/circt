//===- EmitCEmissionPatterns.cpp - EmitC Dialect Emission Patterns --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the emitc dialect.
//
//===----------------------------------------------------------------------===//

#include "EmitCEmissionPatterns.h"
#include "../EmissionPrinter.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir::emitc;
using namespace circt;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit emitc.include operations.
struct IncludeEmitter : OpEmissionPattern<IncludeOp> {
  using OpEmissionPattern::OpEmissionPattern;

  void emitStatement(IncludeOp op, EmissionPrinter &p) override {
    p << "#include " << (op.getIsStandardInclude() ? "<" : "\"")
      << op.getInclude() << (op.getIsStandardInclude() ? ">" : "\"") << "\n";
  }
};

/// Emit emitc.address_of operations.
struct AddressOfOpEmitter : OpEmissionPattern<AddressOfOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<AddressOfOp>())
      return Precedence::ADDRESS_OF;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto addressOfOp = value.getDefiningOp<AddressOfOp>();
    p << "&";
    p.getInlinable(addressOfOp.getReference())
        .emitWithParensOnLowerPrecedence(Precedence::ADDRESS_OF);
  }
};

/// Emit emitc.dereference operations.
struct DereferenceOpEmitter : OpEmissionPattern<DereferenceOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    // 	`dereference` has the same precedence as `address_of`.
    if (value.getDefiningOp<DereferenceOp>())
      return Precedence::ADDRESS_OF;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto dereferenceOp = value.getDefiningOp<DereferenceOp>();
    p << "*";
    p.getInlinable(dereferenceOp.getPointer())
        .emitWithParensOnLowerPrecedence(Precedence::ADDRESS_OF);
  }
};

/// Emit emitc.call operations. Only calls with 0 or 1 results are supported.
/// Calls with no result are emitted as statements whereas calls with exactly
/// one result are always inlined no matter whether it is a pure function or has
/// side effects. To make sure that calls with side effects are not reordered
/// with interferring operations, a pre-pass has to emit VariableOp operations
/// with the result of the call as initial value.
class CallOpEmitter : public OpEmissionPattern<CallOpaqueOp> {
public:
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (auto callOp = value.getDefiningOp<CallOpaqueOp>()) {
      // TODO: template arguments not supported for now.
      if (callOp->getNumResults() == 1 && !callOp.getTemplateArgs())
        return Precedence::FUNCTION_CALL;
    }
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    printCallOp(value.getDefiningOp<CallOpaqueOp>(), p);
  }

  bool matchStatement(Operation *op) override {
    // TODO: template arguments not supported for now.
    if (auto callOp = dyn_cast<CallOpaqueOp>(op))
      return callOp->getNumResults() <= 1 && !callOp.getTemplateArgs();
    return false;
  }

  void emitStatement(CallOpaqueOp callOp, EmissionPrinter &p) override {
    if (callOp->getNumResults() != 0)
      return;

    printCallOp(callOp, p);
    p << ";\n";
  }

private:
  void printCallOp(CallOpaqueOp callOp, EmissionPrinter &p) {
    p << callOp.getCallee();

    p << "(";

    if (!callOp.getArgs()) {
      llvm::interleaveComma(callOp.getOperands(), p, [&](Value operand) {
        p.getInlinable(operand).emitWithParensOnLowerPrecedence(
            Precedence::COMMA);
      });
    } else {
      llvm::interleaveComma(callOp.getArgs().value(), p, [&](Attribute attr) {
        if (auto idx = dyn_cast<IntegerAttr>(attr)) {
          if (isa<IndexType>(idx.getType())) {
            p.getInlinable(callOp.getOperands()[idx.getInt()])
                .emitWithParensOnLowerPrecedence(Precedence::COMMA);
            return;
          }
        }

        p.emitAttr(attr);
      });
    }

    p << ")";
  }
};

/// Emit emitc.cast operations.
struct CastOpEmitter : OpEmissionPattern<CastOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<CastOp>())
      return Precedence::CAST;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto castOp = value.getDefiningOp<CastOp>();
    p << "(";
    p.emitType(castOp.getDest().getType());
    p << ") ";
    p.getInlinable(castOp.getSource())
        .emitWithParensOnLowerPrecedence(Precedence::CAST);
  }
};

/// Emit emitc.constant operations.
struct ConstantEmitter : OpEmissionPattern<ConstantOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<ConstantOp>())
      return Precedence::LIT;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    p.emitAttr(value.getDefiningOp<ConstantOp>().getValue());
  }
};

/// Emit an emitc.variable operation.
struct VariableEmitter : OpEmissionPattern<VariableOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (auto varOp = value.getDefiningOp<VariableOp>()) {
      if (!varOp->getAttrOfType<StringAttr>("name"))
        return {};
      return Precedence::VAR;
    }
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    p << value.getDefiningOp()->getAttrOfType<StringAttr>("name").getValue();
  }

  void emitStatement(VariableOp op, EmissionPrinter &p) override {
    p.emitType(op.getResult().getType());
    p << " " << op->getAttrOfType<StringAttr>("name").getValue();

    if (op.getValue()) {
      p << " = ";
      p.emitAttr(op.getValue());
    }

    p << ";\n";
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Type emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit an emitc.opaque type by just printing the contained string without
/// quotation marks.
struct OpaqueTypeEmitter : TypeEmissionPattern<OpaqueType> {
  void emitType(OpaqueType type, EmissionPrinter &p) override {
    p << type.getValue();
  }
};

/// Emit an emitc.ptr type.
struct PointerTypeEmitter : TypeEmissionPattern<PointerType> {
  void emitType(PointerType type, EmissionPrinter &p) override {
    p.emitType(type.getPointee());
    p << "*";
  }
};

/// Emit an emitc.ptr type.
struct LValueTypeEmitter : TypeEmissionPattern<LValueType> {
  void emitType(LValueType type, EmissionPrinter &p) override {
    p.emitType(type.getValueType());
  }
};
} // namespace

namespace {

/// Emit an emitc.opaque attribute by just printing the contained string without
/// quotation marks.
struct OpaqueAttrEmitter : AttrEmissionPattern<OpaqueAttr> {
  void emitAttr(OpaqueAttr attr, EmissionPrinter &p) override {
    p << attr.getValue();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateEmitCOpEmitters(
    OpEmissionPatternSet &patterns, MLIRContext *context) {
  patterns.add<IncludeEmitter, AddressOfOpEmitter, DereferenceOpEmitter,
               CallOpEmitter, CastOpEmitter, ConstantEmitter, VariableEmitter>(
      context);
}

void circt::ExportSystemC::populateEmitCTypeEmitters(
    TypeEmissionPatternSet &patterns) {
  patterns.add<OpaqueTypeEmitter, PointerTypeEmitter, LValueTypeEmitter>();
}

void circt::ExportSystemC::populateEmitCAttrEmitters(
    AttrEmissionPatternSet &patterns) {
  patterns.add<OpaqueAttrEmitter>();
}
