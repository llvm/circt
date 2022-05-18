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

void FIRRTLDialect::initialize() {
  // Register types and attributes.
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/FIRRTL/FIRRTL.cpp.inc"
      >();
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

    assert((!type.cast<IntType>().hasWidth() ||
            (unsigned)type.cast<IntType>().getWidthOrSentinel() ==
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
