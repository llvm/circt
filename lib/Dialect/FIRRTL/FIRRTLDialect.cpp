//===- FIRRTLDialect.cpp - Implement the FIRRTL dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// If the specified attribute set contains the firrtl.name attribute, return it.
StringAttr firrtl::getFIRRTLNameAttr(ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs) {
    // FIXME: We currently use firrtl.name instead of name because this makes
    // the FunctionLike handling in MLIR core happier.  It otherwise doesn't
    // allow attributes on module parameters.
    if (argAttr.first != "firrtl.name")
      continue;

    return argAttr.second.dyn_cast<StringAttr>();
  }

  return StringAttr();
}

namespace {

// We implement the OpAsmDialectInterface so that FIRRTL dialect operations
// automatically interpret the name attribute on function arguments and
// on operations as their SSA name.
struct FIRRTLOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const override {
    if (auto instance = dyn_cast<InstanceOp>(op)) {
      StringRef base;
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        base = nameAttr.getValue();
      if (base.empty())
        base = "inst";

      for (size_t i = 0, e = op->getNumResults(); i != e; ++i) {
        setNameFn(instance.getResult(i),
                  (base + "_" + instance.getPortNameStr(i)).str());
      }
      return;
    }

    if (auto memory = dyn_cast<MemOp>(op)) {
      StringRef base;
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        base = nameAttr.getValue();
      if (base.empty())
        base = "mem";

      for (size_t i = 0, e = op->getNumResults(); i != e; ++i) {
        setNameFn(memory.getResult(i),
                  (base + "_" + memory.getPortNameStr(i)).str());
      }
      return;
    }

    // Many firrtl dialect operations have an optional 'name' attribute.  If
    // present, use it.
    if (op->getNumResults() == 1)
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        setNameFn(op->getResult(0), nameAttr.getValue());

    // For constants in particular, propagate the value into the result name to
    // make it easier to read the IR.
    if (auto constant = dyn_cast<ConstantOp>(op)) {
      auto intTy = constant.getType().dyn_cast<IntType>();

      // Otherwise, build a complex name with the value and type.
      SmallString<32> specialNameBuffer;
      llvm::raw_svector_ostream specialName(specialNameBuffer);
      specialName << 'c';
      if (intTy) {
        constant.value().print(specialName, /*isSigned:*/ intTy.isSigned());

        specialName << (intTy.isSigned() ? "_si" : "_ui");
        auto width = intTy.getWidthOrSentinel();
        if (width != -1)
          specialName << width;
      } else {
        constant.value().print(specialName, /*isSigned:*/ false);
      }
      setNameFn(constant.getResult(), specialName.str());
    }
  }

  /// Get a special name to use when printing the entry block arguments of the
  /// region contained by an operation in this dialect.
  void getAsmBlockArgumentNames(Block *block,
                                OpAsmSetValueNameFn setNameFn) const override {
    // Check to see if the operation containing the arguments has 'firrtl.name'
    // attributes for them.  If so, use that as the name.
    auto *parentOp = block->getParentOp();

    for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
      // Scan for a 'firrtl.name' attribute.
      if (auto str = getFIRRTLNameAttr(impl::getArgAttrs(parentOp, i)))
        setNameFn(block->getArgument(i), str.getValue());
    }
  }
};
} // end anonymous namespace

FIRRTLDialect::FIRRTLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              ::mlir::TypeID::get<FIRRTLDialect>()) {

  // Register types.
  addTypes<SIntType, UIntType, ClockType, ResetType, AsyncResetType, AnalogType,
           // Derived Types
           FlipType, BundleType, FVectorType>();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/FIRRTL/FIRRTL.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<FIRRTLOpAsmDialectInterface>();
}

FIRRTLDialect::~FIRRTLDialect() {}

void FIRRTLDialect::printType(Type type, DialectAsmPrinter &os) const {
  type.cast<FIRRTLType>().print(os.getStream());
}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *FIRRTLDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  // Integer constants.
  if (auto intType = type.dyn_cast<IntType>())
    if (auto attrValue = value.dyn_cast<IntegerAttr>())
      return builder.create<ConstantOp>(loc, type, attrValue);

  return nullptr;
}

// Provide implementations for the enums we use.
#include "circt/Dialect/FIRRTL/FIRRTLEnums.cpp.inc"
