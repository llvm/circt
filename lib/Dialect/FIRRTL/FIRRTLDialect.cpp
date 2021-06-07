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

    // TODO: implement subindex op.
    if (auto subfieldOp = dyn_cast<SubfieldOp>(op)) {
      value = subfieldOp.input();
      // Strip any flip wrapping the bundle type.
      auto type = value.getType();
      if (auto flipType = type.dyn_cast<FlipType>())
        type = flipType.getElementType();
      auto bundleType = type.cast<BundleType>();
      auto index =
          bundleType.getElementIndex(subfieldOp.fieldname()).getValue();
      // Rebase the current index on the parent field's index.
      id += bundleType.getFieldID(index);
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
    SmallVector<ModulePortInfo> ports = module.getPorts();
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

  SmallString<64> name;
  auto value = fieldRef.getValue();
  getDeclName(value, name);

  auto type = value.getType();
  auto localID = fieldRef.getFieldID();
  while (localID) {
    // Strip off the flip type if there is one.
    if (auto flipType = type.dyn_cast<FlipType>())
      type = flipType.getElementType();
    // TODO: support vector types.
    auto bundleType = type.cast<BundleType>();
    auto index = bundleType.getIndexForFieldID(localID);
    // Add the current field string, and recurse into a subfield.
    auto &element = bundleType.getElements()[index];
    name += ".";
    name += element.name.getValue();
    type = element.type;
    // Get a field localID for the nested bundle.
    localID = localID - bundleType.getFieldID(index);
  }

  return name.str().str();
}

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// If the specified module contains the portNames attribute, return it.
ArrayAttr firrtl::getModulePortNames(Operation *module) {
  return module->getAttrOfType<ArrayAttr>("portNames");
}

// If the specified module contains the portDirections attribute, return it.
mlir::IntegerAttr firrtl::getModulePortDirections(Operation *module) {
  return module->getAttrOfType<mlir::IntegerAttr>(direction::attrKey);
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
    }
  }

  /// Get a special name to use when printing the entry block arguments of the
  /// region contained by an operation in this dialect.
  void getAsmBlockArgumentNames(Block *block,
                                OpAsmSetValueNameFn setNameFn) const override {
    // Check to see if the operation containing the arguments has 'firrtl.name'
    // attributes for them.  If so, use that as the name.
    auto *parentOp = block->getParentOp();
    auto argAttr = getModulePortNames(parentOp);

    // Do not crash on invalid IR.
    if (!argAttr || argAttr.size() != block->getNumArguments())
      return;

    for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
      auto str = argAttr[i].cast<StringAttr>().getValue();
      if (!str.empty())
        setNameFn(block->getArgument(i), str);
    }
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
  if (auto attrValue = value.dyn_cast<IntegerAttr>()) {
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
