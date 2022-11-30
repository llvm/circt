//===- FIRRTLUtils.cpp - FIRRTL IR Utilities --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various utilties to help generate and process FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

void circt::firrtl::emitConnect(OpBuilder &builder, Location loc, Value dst,
                                Value src) {
  ImplicitLocOpBuilder locBuilder(loc, builder.getInsertionBlock(),
                                  builder.getInsertionPoint());
  emitConnect(locBuilder, dst, src);
  builder.restoreInsertionPoint(locBuilder.saveInsertionPoint());
}

/// Emit a connect between two values.
void circt::firrtl::emitConnect(ImplicitLocOpBuilder &builder, Value dst,
                                Value src) {
  auto dstFType = dst.getType().cast<FIRRTLType>();
  auto srcFType = src.getType().cast<FIRRTLType>();
  auto dstType = dstFType.dyn_cast<FIRRTLBaseType>();
  auto srcType = srcFType.dyn_cast<FIRRTLBaseType>();

  // If the types are the exact same we can just connect them.
  if (dstFType == srcFType) {
    // Strict connect does not allow uninferred widths.
    if (dstType && dstType.hasUninferredWidth())
      builder.create<ConnectOp>(dst, src);
    else
      builder.create<StrictConnectOp>(dst, src);
    return;
  }

  // Non-base types don't need special handling.
  if (!srcType || !dstType) {
    builder.create<ConnectOp>(dst, src);
    return;
  }

  if (auto dstBundle = dstType.dyn_cast<BundleType>()) {
    // Connect all the bundle elements pairwise.
    auto numElements = dstBundle.getNumElements();
    // Check if we are trying to create an illegal connect - just create the
    // connect and let the verifier catch it.
    auto srcBundle = srcType.dyn_cast<BundleType>();
    if (!srcBundle || numElements != srcBundle.getNumElements()) {
      builder.create<ConnectOp>(dst, src);
      return;
    }
    for (size_t i = 0; i < numElements; ++i) {
      auto dstField = builder.create<SubfieldOp>(dst, i);
      auto srcField = builder.create<SubfieldOp>(src, i);
      if (dstBundle.getElement(i).isFlip)
        std::swap(dstField, srcField);
      emitConnect(builder, dstField, srcField);
    }
    return;
  }

  if (auto dstVector = dstType.dyn_cast<FVectorType>()) {
    // Connect all the vector elements pairwise.
    auto numElements = dstVector.getNumElements();
    // Check if we are trying to create an illegal connect - just create the
    // connect and let the verifier catch it.
    auto srcVector = srcType.dyn_cast<FVectorType>();
    if (!srcVector || numElements != srcVector.getNumElements()) {
      builder.create<ConnectOp>(dst, src);
      return;
    }
    for (size_t i = 0; i < numElements; ++i) {
      auto dstField = builder.create<SubindexOp>(dst, i);
      auto srcField = builder.create<SubindexOp>(src, i);
      emitConnect(builder, dstField, srcField);
    }
    return;
  }

  // Handle ground types with possibly uninferred widths.
  auto dstWidth = dstType.getBitWidthOrSentinel();
  auto srcWidth = srcType.getBitWidthOrSentinel();
  if (dstWidth < 0 || srcWidth < 0) {
    // If one of these types has an uninferred width, we connect them with a
    // regular connect operation.
    builder.create<ConnectOp>(dst, src);
    return;
  }

  // The source must be extended or truncated.
  if (dstWidth < srcWidth) {
    // firrtl.tail always returns uint even for sint operands.
    IntType tmpType = dstType.cast<IntType>();
    if (tmpType.isSigned())
      tmpType = UIntType::get(dstType.getContext(), dstWidth);
    src = builder.create<TailPrimOp>(tmpType, src, srcWidth - dstWidth);
    // Insert the cast back to signed if needed.
    if (tmpType != dstType)
      src = builder.create<AsSIntPrimOp>(dstType, src);
  } else if (srcWidth < dstWidth) {
    // Need to extend arg.
    src = builder.create<PadPrimOp>(src, dstWidth);
  }

  // Strict connect requires the types to be completely equal, including
  // connecting uint<1> to abstract reset types.
  if (dstType == src.getType())
    builder.create<StrictConnectOp>(dst, src);
  else
    builder.create<ConnectOp>(dst, src);
}

IntegerAttr circt::firrtl::getIntAttr(Type type, const APInt &value) {
  auto intType = type.cast<IntType>();
  assert((!intType.hasWidth() ||
          (unsigned)intType.getWidthOrSentinel() == value.getBitWidth()) &&
         "value / type width mismatch");
  auto intSign =
      intType.isSigned() ? IntegerType::Signed : IntegerType::Unsigned;
  auto attrType =
      IntegerType::get(type.getContext(), value.getBitWidth(), intSign);
  return IntegerAttr::get(attrType, value);
}

/// Return an IntegerAttr filled with zeros for the specified FIRRTL integer
/// type. This handles both the known width and unknown width case.
IntegerAttr circt::firrtl::getIntZerosAttr(Type type) {
  int32_t width = abs(type.cast<IntType>().getWidthOrSentinel());
  return getIntAttr(type, APInt(width, 0));
}

/// Return an IntegerAttr filled with ones for the specified FIRRTL integer
/// type. This handles both the known width and unknown width case.
IntegerAttr circt::firrtl::getIntOnesAttr(Type type) {
  int32_t width = abs(type.cast<IntType>().getWidthOrSentinel());
  return getIntAttr(type, APInt(width, -1));
}

/// Return the value that drives another FIRRTL value within module scope.  Only
/// look backwards through one connection.  This is intended to be used in
/// situations where you only need to look at the most recent connect, e.g., to
/// know if a wire has been driven to a constant.  Return null if no driver via
/// a connect was found.
Value circt::firrtl::getDriverFromConnect(Value val) {
  for (auto *user : val.getUsers()) {
    if (auto connect = dyn_cast<FConnectLike>(user)) {
      if (connect.getDest() != val)
        continue;
      return connect.getSrc();
    }
  }
  return nullptr;
}

Value circt::firrtl::getModuleScopedDriver(Value val, bool lookThroughWires,
                                           bool lookThroughNodes,
                                           bool lookThroughCasts) {
  // Update `val` to the source of the connection driving `thisVal`.  This walks
  // backwards across users to find the first connection and updates `val` to
  // the source.  This assumes that only one connect is driving `thisVal`, i.e.,
  // this pass runs after `ExpandWhens`.
  auto updateVal = [&](Value thisVal) {
    for (auto *user : thisVal.getUsers()) {
      if (auto connect = dyn_cast<FConnectLike>(user)) {
        if (connect.getDest() != val)
          continue;
        val = connect.getSrc();
        return;
      }
    }
    val = nullptr;
    return;
  };

  while (val) {
    // The value is a port.
    if (auto blockArg = val.dyn_cast<BlockArgument>()) {
      FModuleOp op = cast<FModuleOp>(val.getParentBlock()->getParentOp());
      auto direction = op.getPortDirection(blockArg.getArgNumber());
      // Base case: this is one of the module's input ports.
      if (direction == Direction::In)
        return blockArg;
      updateVal(blockArg);
      continue;
    }

    auto *op = val.getDefiningOp();

    // The value is an instance port.
    if (auto inst = dyn_cast<InstanceOp>(op)) {
      auto resultNo = val.cast<OpResult>().getResultNumber();
      // Base case: this is an instance's output port.
      if (inst.getPortDirection(resultNo) == Direction::Out)
        return inst.getResult(resultNo);
      updateVal(val);
      continue;
    }

    // If told to look through wires, continue from the driver of the wire.
    if (lookThroughWires && isa<WireOp>(op)) {
      updateVal(op->getResult(0));
      continue;
    }

    // If told to look through nodes, continue from the node input.
    if (lookThroughNodes && isa<NodeOp>(op)) {
      val = cast<NodeOp>(op).getInput();
      continue;
    }

    if (lookThroughCasts &&
        isa<AsUIntPrimOp, AsSIntPrimOp, AsClockPrimOp, AsAsyncResetPrimOp>(
            op)) {
      val = op->getOperand(0);
      continue;
    }

    // Look through unary ops generated by emitConnect
    if (isa<PadPrimOp, TailPrimOp>(op)) {
      val = op->getOperand(0);
      continue;
    }

    // Base case: this is a constant/invalid or primop.
    //
    // TODO: If needed, this could be modified to look through unary ops which
    // have an unambiguous single driver.  This should only be added if a need
    // arises for it.
    break;
  };
  return val;
}

bool circt::firrtl::walkDrivers(Value val, bool lookThroughWires,
                                bool lookThroughNodes, bool lookThroughCasts,
                                WalkDriverCallback callback) {
  // TODO: what do we want to happen when there are flips in the type? Do we
  // want to filter out fields which have reverse flow?
  assert(val.getType().cast<FIRRTLBaseType>().isPassive() &&
         "this code was not tested with flips");

  // This method keeps a stack of wires (or ports) and subfields of those that
  // it still has to process.  It keeps track of which fields in the
  // destination are attached to which fields of the source, as well as which
  // subfield of the source we are currently investigating.  The fieldID is
  // used to filter which subfields of the current operation which we should
  // visit. As an example, the src might be an aggregate wire, but the current
  // value might be a subfield of that wire. The `src` FieldRef will represent
  // all subaccesses to the target, but `fieldID` for the current op only needs
  // to represent the all subaccesses between the current op and the target.
  struct StackElement {
    StackElement(FieldRef dst, FieldRef src, Value current, unsigned fieldID)
        : dst(dst), src(src), current(current), it(current.user_begin()),
          fieldID(fieldID) {}
    // The elements of the destination that this refers to.
    FieldRef dst;
    // The elements of the source that this refers to.
    FieldRef src;

    // These next fields are tied to the value we are currently iterating. This
    // is used so we can check if a connect op is reading or driving from this
    // value.
    Value current;
    // An iterator of the users of the current value. An end() iterator can be
    // constructed from the `current` value.
    Value::user_iterator it;
    // A filter for which fields of the current value we care about.
    unsigned fieldID;
  };
  SmallVector<StackElement> workStack;

  // Helper to add record a new wire to be processed in the worklist.  This will
  // add the wire itself to the worklist, which will lead to all subaccesses
  // being eventually processed as well.
  auto addToWorklist = [&](FieldRef dst, FieldRef src) {
    auto value = src.getValue();
    workStack.emplace_back(dst, src, value, src.getFieldID());
  };

  // Create an initial fieldRef from the input value.  As a starting state, the
  // dst and src are the same value.
  auto original = getFieldRefFromValue(val);
  auto fieldRef = original;

  // This loop wraps the worklist, which processes wires. Initially the worklist
  // is empty.
  while (true) {
    // This loop looks through simple operations like casts and nodes.  If it
    // encounters a wire it will stop and add the wire to the worklist.
    while (true) {
      auto val = fieldRef.getValue();

      // The value is a port.
      if (auto blockArg = val.dyn_cast<BlockArgument>()) {
        FModuleOp op = cast<FModuleOp>(val.getParentBlock()->getParentOp());
        auto direction = op.getPortDirection(blockArg.getArgNumber());
        // Base case: this is one of the module's input ports.
        if (direction == Direction::In) {
          if (!callback(original, fieldRef))
            return false;
          break;
        }
        addToWorklist(original, fieldRef);
        break;
      }

      auto *op = val.getDefiningOp();

      // The value is an instance port.
      if (auto inst = dyn_cast<InstanceOp>(op)) {
        auto resultNo = val.cast<OpResult>().getResultNumber();
        // Base case: this is an instance's output port.
        if (inst.getPortDirection(resultNo) == Direction::Out) {
          if (!callback(original, fieldRef))
            return false;
          break;
        }
        addToWorklist(original, fieldRef);
        break;
      }

      // If told to look through wires, continue from the driver of the wire.
      if (lookThroughWires && isa<WireOp>(op)) {
        addToWorklist(original, fieldRef);
        break;
      }

      // If told to look through nodes, continue from the node input.
      if (lookThroughNodes && isa<NodeOp>(op)) {
        auto input = cast<NodeOp>(op).getInput();
        auto next = getFieldRefFromValue(input);
        fieldRef = next.getSubField(fieldRef.getFieldID());
        continue;
      }

      // If told to look through casts, continue from the cast input.
      if (lookThroughCasts &&
          isa<AsUIntPrimOp, AsSIntPrimOp, AsClockPrimOp, AsAsyncResetPrimOp>(
              op)) {
        auto input = op->getOperand(0);
        auto next = getFieldRefFromValue(input);
        fieldRef = next.getSubField(fieldRef.getFieldID());
        continue;
      }

      // Look through unary ops generated by emitConnect.
      if (isa<PadPrimOp, TailPrimOp>(op)) {
        auto input = op->getOperand(0);
        auto next = getFieldRefFromValue(input);
        fieldRef = next.getSubField(fieldRef.getFieldID());
        continue;
      }

      // Base case: this is a constant/invalid or primop.
      //
      // TODO: If needed, this could be modified to look through unary ops which
      // have an unambiguous single driver.  This should only be added if a need
      // arises for it.
      if (!callback(original, fieldRef))
        return false;
      break;
    }

    // Process the next element on the stack.
    while (true) {
      // If there is nothing left in the workstack, we are done.
      if (workStack.empty())
        return true;
      auto &back = workStack.back();
      auto current = back.current;
      // Pop the current element if we have processed all users.
      if (back.it == current.user_end()) {
        workStack.pop_back();
        continue;
      }

      original = back.dst;
      fieldRef = back.src;
      auto *user = *back.it++;
      auto fieldID = back.fieldID;

      if (auto subfield = dyn_cast<SubfieldOp>(user)) {
        auto bundleType = subfield.getInput().getType().cast<BundleType>();
        auto index = subfield.getFieldIndex();
        auto subID = bundleType.getFieldID(index);
        // If the index of this operation doesn't match the target, skip it.
        if (fieldID && index != bundleType.getIndexForFieldID(fieldID))
          continue;
        auto subRef = fieldRef.getSubField(subID);
        auto subOriginal = original.getSubField(subID);
        auto value = subfield.getResult();
        workStack.emplace_back(subOriginal, subRef, value, fieldID - subID);
      } else if (auto subindex = dyn_cast<SubindexOp>(user)) {
        auto vectorType = subindex.getInput().getType().cast<FVectorType>();
        auto index = subindex.getIndex();
        auto subID = vectorType.getFieldID(index);
        // If the index of this operation doesn't match the target, skip it.
        if (fieldID && index != vectorType.getIndexForFieldID(fieldID))
          continue;
        auto subRef = fieldRef.getSubField(subID);
        auto subOriginal = original.getSubField(subID);
        auto value = subindex.getResult();
        workStack.emplace_back(subOriginal, subRef, value, fieldID - subID);
      } else if (auto connect = dyn_cast<FConnectLike>(user)) {
        // Make sure that this connect is driving the value.
        if (connect.getDest() != current)
          continue;
        // If the value is driven by a connect, we don't have to recurse,
        // just update the current value.
        fieldRef = getFieldRefFromValue(connect.getSrc());
        break;
      }
    }
  }
}

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
      value = subfieldOp.getInput();
      auto bundleType = value.getType().cast<BundleType>();
      // Rebase the current index on the parent field's index.
      id += bundleType.getFieldID(subfieldOp.getFieldIndex());
    } else if (auto subindexOp = dyn_cast<SubindexOp>(op)) {
      value = subindexOp.getInput();
      auto vecType = value.getType().cast<FVectorType>();
      // Rebase the current index on the parent field's index.
      id += vecType.getFieldID(subindexOp.getIndex());
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
        string += op.getName();
        string += ".";
        string +=
            op.getPortName(value.cast<OpResult>().getResultNumber()).getValue();
      })
      .Case<WireOp, RegOp, RegResetOp>(
          [&](auto op) { string += op.getName(); });
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

/// This gets the value targeted by a field id.  If the field id is targeting
/// the value itself, it returns it unchanged. If it is targeting a single field
/// in a aggregate value, such as a bundle or vector, this will create the
/// necessary subaccesses to get the value.
Value circt::firrtl::getValueByFieldID(ImplicitLocOpBuilder builder,
                                       Value value, unsigned fieldID) {
  // When the fieldID hits 0, we've found the target value.
  while (fieldID != 0) {
    auto type = value.getType();
    if (auto bundle = type.dyn_cast<BundleType>()) {
      auto index = bundle.getIndexForFieldID(fieldID);
      value = builder.create<SubfieldOp>(value, index);
      fieldID -= bundle.getFieldID(index);
    } else {
      auto vector = type.cast<FVectorType>();
      auto index = vector.getIndexForFieldID(fieldID);
      value = builder.create<SubindexOp>(value, index);
      fieldID -= vector.getFieldID(index);
    }
  }
  return value;
}

/// Returns an operation's `inner_sym`, adding one if necessary.
StringAttr circt::firrtl::getOrAddInnerSym(
    Operation *op, StringRef nameHint, FModuleOp mod,
    std::function<ModuleNamespace &(FModuleOp)> getNamespace) {
  auto attr = getInnerSymName(op);
  if (attr)
    return attr;
  if (nameHint.empty()) {
    if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
      nameHint = nameAttr.getValue();
    else
      nameHint = "sym";
  }
  auto name = getNamespace(mod).newName(nameHint);
  attr = StringAttr::get(op->getContext(), name);
  op->setAttr("inner_sym", hw::InnerSymAttr::get(attr));
  return attr;
}

/// Obtain an inner reference to an operation, possibly adding an `inner_sym`
/// to that operation.
hw::InnerRefAttr circt::firrtl::getInnerRefTo(
    Operation *op, StringRef nameHint,
    std::function<ModuleNamespace &(FModuleOp)> getNamespace) {
  auto mod = op->getParentOfType<FModuleOp>();
  assert(mod && "must be an operation inside an FModuleOp");
  return hw::InnerRefAttr::get(
      SymbolTable::getSymbolName(mod),
      getOrAddInnerSym(op, nameHint, mod, getNamespace));
}

/// Returns a port's `inner_sym`, adding one if necessary.
StringAttr circt::firrtl::getOrAddInnerSym(
    FModuleLike mod, size_t portIdx, StringRef nameHint,
    std::function<ModuleNamespace &(FModuleLike)> getNamespace) {

  auto attr = mod.getPortSymbolAttr(portIdx);
  if (attr)
    return attr.getSymName();
  if (nameHint.empty()) {
    if (auto name = mod.getPortNameAttr(portIdx))
      nameHint = name;
    else
      nameHint = "sym";
  }
  auto name = getNamespace(mod).newName(nameHint);
  auto sAttr = StringAttr::get(mod.getContext(), name);
  mod.setPortSymbolAttr(portIdx, sAttr);
  return sAttr;
}

/// Obtain an inner reference to a port, possibly adding an `inner_sym`
/// to the port.
hw::InnerRefAttr circt::firrtl::getInnerRefTo(
    FModuleLike mod, size_t portIdx, StringRef nameHint,
    std::function<ModuleNamespace &(FModuleLike)> getNamespace) {
  return hw::InnerRefAttr::get(
      SymbolTable::getSymbolName(mod),
      getOrAddInnerSym(mod, portIdx, nameHint, getNamespace));
}

/// Parse a string that may encode a FIRRTL location into a LocationAttr.
std::pair<bool, Optional<mlir::LocationAttr>>
circt::firrtl::maybeStringToLocation(StringRef spelling, bool skipParsing,
                                     StringAttr &locatorFilenameCache,
                                     FileLineColLoc &fileLineColLocCache,
                                     MLIRContext *context) {
  // The spelling of the token looks something like "@[Decoupled.scala 221:8]".
  if (!spelling.startswith("@[") || !spelling.endswith("]"))
    return {false, None};

  spelling = spelling.drop_front(2).drop_back(1);

  // Decode the locator in "spelling", returning the filename and filling in
  // lineNo and colNo on success.  On failure, this returns an empty filename.
  auto decodeLocator = [&](StringRef input, unsigned &resultLineNo,
                           unsigned &resultColNo) -> StringRef {
    // Split at the last space.
    auto spaceLoc = input.find_last_of(' ');
    if (spaceLoc == StringRef::npos)
      return {};

    auto filename = input.take_front(spaceLoc);
    auto lineAndColumn = input.drop_front(spaceLoc + 1);

    // Decode the line/column.  If the colon is missing, then it will be empty
    // here.
    StringRef lineStr, colStr;
    std::tie(lineStr, colStr) = lineAndColumn.split(':');

    // Decode the line number and the column number if present.
    if (lineStr.getAsInteger(10, resultLineNo))
      return {};
    if (!colStr.empty()) {
      if (colStr.front() != '{') {
        if (colStr.getAsInteger(10, resultColNo))
          return {};
      } else {
        // compound locator, just parse the first part for now
        if (colStr.drop_front().split(',').first.getAsInteger(10, resultColNo))
          return {};
      }
    }
    return filename;
  };

  // Decode the locator spelling, reporting an error if it is malformed.
  unsigned lineNo = 0, columnNo = 0;
  StringRef filename = decodeLocator(spelling, lineNo, columnNo);
  if (filename.empty())
    return {false, None};

  // If info locators are ignored, don't actually apply them.  We still do all
  // the verification above though.
  if (skipParsing)
    return {true, None};

  /// Return an FileLineColLoc for the specified location, but use a bit of
  /// caching to reduce thrasing the MLIRContext.
  auto getFileLineColLoc = [&](StringRef filename, unsigned lineNo,
                               unsigned columnNo) -> FileLineColLoc {
    // Check our single-entry cache for this filename.
    StringAttr filenameId = locatorFilenameCache;
    if (filenameId.str() != filename) {
      // We missed!  Get the right identifier.
      locatorFilenameCache = filenameId = StringAttr::get(context, filename);

      // If we miss in the filename cache, we also miss in the FileLineColLoc
      // cache.
      return fileLineColLocCache =
                 FileLineColLoc::get(filenameId, lineNo, columnNo);
    }

    // If we hit the filename cache, check the FileLineColLoc cache.
    auto result = fileLineColLocCache;
    if (result && result.getLine() == lineNo && result.getColumn() == columnNo)
      return result;

    return fileLineColLocCache =
               FileLineColLoc::get(filenameId, lineNo, columnNo);
  };

  // Compound locators will be combined with spaces, like:
  //  @[Foo.scala 123:4 Bar.scala 309:14]
  // and at this point will be parsed as a-long-string-with-two-spaces at
  // 309:14.   We'd like to parse this into two things and represent it as an
  // MLIR fused locator, but we want to be conservatively safe for filenames
  // that have a space in it.  As such, we are careful to make sure we can
  // decode the filename/loc of the result.  If so, we accumulate results,
  // backward, in this vector.
  SmallVector<Location> extraLocs;
  auto spaceLoc = filename.find_last_of(' ');
  while (spaceLoc != StringRef::npos) {
    // Try decoding the thing before the space.  Validates that there is another
    // space and that the file/line can be decoded in that substring.
    unsigned nextLineNo = 0, nextColumnNo = 0;
    auto nextFilename =
        decodeLocator(filename.take_front(spaceLoc), nextLineNo, nextColumnNo);

    // On failure we didn't have a joined locator.
    if (nextFilename.empty())
      break;

    // On success, remember what we already parsed (Bar.Scala / 309:14), and
    // move on to the next chunk.
    auto loc =
        getFileLineColLoc(filename.drop_front(spaceLoc + 1), lineNo, columnNo);
    extraLocs.push_back(loc);
    filename = nextFilename;
    lineNo = nextLineNo;
    columnNo = nextColumnNo;
    spaceLoc = filename.find_last_of(' ');
  }

  mlir::LocationAttr result = getFileLineColLoc(filename, lineNo, columnNo);
  if (!extraLocs.empty()) {
    extraLocs.push_back(result);
    std::reverse(extraLocs.begin(), extraLocs.end());
    result = FusedLoc::get(context, extraLocs);
  }
  return {true, result};
}
