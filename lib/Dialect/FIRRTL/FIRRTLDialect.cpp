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
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// FieldRef helpers
//===----------------------------------------------------------------------===//

FieldRef circt::firrtl::getFieldRefFromValue(Value value) {
  // This code walks upwards from the subfield and calculates the field ID at
  // each level. At each stage, it must take the current id, and re-index it as
  // a nested bundle under the parent field.. This is accomplished by using the
  // parent field's ID as a base, and adding the field ID of the child.
  unsigned id = 0;
  while (value) {
    Operation *op = value.getDefiningOp();

    // If this is a block argument, we are done.
    if (!op)
      break;

    if (auto subfieldOp = dyn_cast<SubfieldOp>(op)) {
      value = subfieldOp.input();
      auto bundleType = value.getType().cast<BundleType>();
      // Rebase the current index on the parent field's index.
      id += bundleType.getFieldID(subfieldOp.fieldIndex());
    } else if (auto subindexOp = dyn_cast<SubindexOp>(op)) {
      value = subindexOp.input();
      auto vecType = value.getType().cast<FVectorType>();
      // Rebase the current index on the parent field's index.
      id += vecType.getFieldID(subindexOp.index());
    } else {
      break;
    }
  }
  return {value, id};
}

/// Get the string name of a value which is a direct child of a declaration op.
static void getDeclName(Value value, SmallString<64> &string) {
  if (auto arg = value.dyn_cast<BlockArgument>()) {
    // Get the module ports and get the name.
    auto module = cast<FModuleOp>(arg.getOwner()->getParentOp());
    SmallVector<PortInfo> ports = module.getPorts();
    string += ports[arg.getArgNumber()].name.getValue();
    return;
  }

  auto *op = value.getDefiningOp();
  TypeSwitch<Operation *>(op)
      .Case<InstanceOp, MemOp>([&](auto op) {
        string += op.name();
        string += ".";
        string +=
            op.getPortName(value.cast<OpResult>().getResultNumber()).getValue();
      })
      .Case<WireOp, RegOp, RegResetOp>([&](auto op) { string += op.name(); });
}

std::string circt::firrtl::getFieldName(const FieldRef &fieldRef) {
  bool rootKnown;
  return getFieldName(fieldRef, rootKnown);
}

std::string circt::firrtl::getFieldName(const FieldRef &fieldRef,
                                        bool &rootKnown) {
  SmallString<64> name;
  auto value = fieldRef.getValue();
  getDeclName(value, name);
  rootKnown = !name.empty();

  auto type = value.getType();
  auto localID = fieldRef.getFieldID();
  while (localID) {
    if (auto bundleType = type.dyn_cast<BundleType>()) {
      auto index = bundleType.getIndexForFieldID(localID);
      // Add the current field string, and recurse into a subfield.
      auto &element = bundleType.getElements()[index];
      if (!name.empty())
        name += ".";
      name += element.name.getValue();
      // Recurse in to the element type.
      type = element.type;
      localID = localID - bundleType.getFieldID(index);
    } else if (auto vecType = type.dyn_cast<FVectorType>()) {
      auto index = vecType.getIndexForFieldID(localID);
      name += "[";
      name += std::to_string(index);
      name += "]";
      // Recurse in to the element type.
      type = vecType.getElementType();
      localID = localID - vecType.getFieldID(index);
    } else {
      // If we reach here, the field ref is pointing inside some aggregate type
      // that isn't a bundle or a vector. If the type is a ground type, then the
      // localID should be 0 at this point, and we should have broken from the
      // loop.
      llvm_unreachable("unsupported type");
    }
  }

  return name.str().str();
}

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

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
      return;
    }

    if (auto specialConstant = dyn_cast<SpecialConstantOp>(op)) {
      SmallString<32> specialNameBuffer;
      llvm::raw_svector_ostream specialName(specialNameBuffer);
      specialName << 'c';
      specialName << static_cast<unsigned>(specialConstant.value());
      auto type = specialConstant.getType();
      if (type.isa<ClockType>()) {
        specialName << "_clock";
      } else if (type.isa<ResetType>()) {
        specialName << "_reset";
      } else if (type.isa<AsyncResetType>()) {
        specialName << "_asyncreset";
      }
      setNameFn(specialConstant.getResult(), specialName.str());
      return;
    }

    // Set invalid values to have a distinct name.
    if (auto invalid = dyn_cast<InvalidValueOp>(op)) {
      std::string name;
      if (auto ty = invalid.getType().dyn_cast<IntType>()) {
        const char *base = ty.isSigned() ? "invalid_si" : "invalid_ui";
        auto width = ty.getWidthOrSentinel();
        if (width == -1)
          name = base;
        else
          name = (Twine(base) + Twine(width)).str();
      } else if (auto ty = invalid.getType().dyn_cast<AnalogType>()) {
        auto width = ty.getWidthOrSentinel();
        if (width == -1)
          name = "invalid_analog";
        else
          name = ("invalid_analog" + Twine(width)).str();
      } else if (invalid.getType().isa<AsyncResetType>())
        name = "invalid_asyncreset";
      else if (invalid.getType().isa<ResetType>())
        name = "invalid_reset";
      else if (invalid.getType().isa<ClockType>())
        name = "invalid_clock";
      else
        name = "invalid";

      setNameFn(invalid.getResult(), name);
      return;
    }

    // Many firrtl dialect operations have an optional 'name' attribute.  If
    // present, use it.
    if (op->getNumResults() == 1)
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        setNameFn(op->getResult(0), nameAttr.getValue());
  }
};

} // end anonymous namespace

void FIRRTLDialect::initialize() {
  // Register types and attributes.
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/FIRRTL/FIRRTL.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<FIRRTLOpAsmDialectInterface>();
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

  // Boolean constants. Boolean attributes are always a special constant type
  // like ClockType and ResetType.  Since BoolAttrs are also IntegerAttrs, its
  // important that this goes first.
  if (auto attrValue = value.dyn_cast<BoolAttr>()) {
    assert((type.isa<ClockType>() || type.isa<AsyncResetType>() ||
            type.isa<ResetType>()) &&
           "BoolAttrs can only be materialized for special constant types.");
    return builder.create<SpecialConstantOp>(loc, type, attrValue);
  }

  // Integer constants.
  if (auto attrValue = value.dyn_cast<IntegerAttr>()) {
    // Integer attributes (ui1) might still be special constant types.
    if (attrValue.getValue().getBitWidth() == 1 &&
        (type.isa<ClockType>() || type.isa<AsyncResetType>() ||
         type.isa<ResetType>()))
      return builder.create<SpecialConstantOp>(
          loc, type,
          builder.getBoolAttr(attrValue.getValue().isAllOnesValue()));

    auto intType = type.cast<IntType>();
    assert((!intType.hasWidth() || (unsigned)intType.getWidthOrSentinel() ==
                                       attrValue.getValue().getBitWidth()) &&
           "type/value width mismatch materializing constant");
    return builder.create<ConstantOp>(loc, type, attrValue);
  }

  // InvalidValue constants.
  if (auto invalidValue = value.dyn_cast<InvalidValueAttr>())
    return builder.create<InvalidValueOp>(loc, type);

  return nullptr;
}

// Provide implementations for the enums we use.
#include "circt/Dialect/FIRRTL/FIRRTLEnums.cpp.inc"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.cpp.inc"
