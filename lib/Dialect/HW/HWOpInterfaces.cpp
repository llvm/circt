//===- HWOpInterfaces.cpp - Implement the HW op interfaces ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the HW operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypeInterfaces.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

using namespace circt;
using namespace hw;

LogicalResult hw::verifyInnerSymAttr(InnerSymbolOpInterface op) {
  auto innerSym = op.getInnerSymAttr();
  // If does not have any inner sym then ignore.
  if (!innerSym)
    return success();

  if (!op.supportsPerFieldSymbols()) {
    // The inner sym can only be specified on fieldID=0.
    if (innerSym.size() > 1 || !innerSym.getSymName()) {
      op->emitOpError("does not support per-field inner symbols");
      return failure();
    }
    return success();
  }

  auto result = op.getTargetResult();
  // If op supports per-field symbols, but does not have a target result,
  // its up to the operation to verify itself.
  // (there are no uses for this presently, but be open to this anyway.)
  if (!result)
    return success();
  auto resultType = result.getType().dyn_cast<FieldIDTypeInterface>();
  // If this type doesn't implement the FieldIDTypeInterface, then there is
  // nothing additional we can check.
  if (!resultType)
    return success();
  auto maxFields = resultType.getMaxFieldID();
  llvm::SmallBitVector indices(maxFields + 1);
  llvm::SmallPtrSet<Attribute, 8> symNames;
  // Ensure fieldID and symbol names are unique.
  auto uniqSyms = [&](InnerSymPropertiesAttr p) {
    if (maxFields < p.getFieldID()) {
      op->emitOpError("field id:'" + Twine(p.getFieldID()) +
                      "' is greater than the maximum field id:'" +
                      Twine(maxFields) + "'");
      return false;
    }
    if (indices.test(p.getFieldID())) {
      op->emitOpError("cannot assign multiple symbol names to the field id:'" +
                      Twine(p.getFieldID()) + "'");
      return false;
    }
    indices.set(p.getFieldID());
    auto it = symNames.insert(p.getName());
    if (!it.second) {
      op->emitOpError("cannot reuse symbol name:'" + p.getName().getValue() +
                      "'");
      return false;
    }
    return true;
  };

  if (!llvm::all_of(innerSym.getProps(), uniqSyms))
    return failure();

  return success();
}



////////////////////////////////////////////////////////////////////////////////
// HWModuleLike Implementation helpers
////////////////////////////////////////////////////////////////////////////////

static bool isEmptyAttrDict(Attribute attr) {
  return llvm::cast<DictionaryAttr>(attr).empty();
}

/// Get either the argument or result attributes array.
template <bool isArg>
static ArrayAttr getArgResAttrs(HWModuleLike op) {
  if constexpr (isArg)
    return op.getArgAttrsAttr();
  else
    return op.getResAttrsAttr();
}

/// Set either the argument or result attributes array.
template <bool isArg>
static void setArgResAttrs(HWModuleLike op, ArrayAttr attrs) {
  if constexpr (isArg)
    op.setArgAttrsAttr(attrs);
  else
    op.setResAttrsAttr(attrs);
}

/// Erase either the argument or result attributes array.
template <bool isArg>
static void removeArgResAttrs(HWModuleLike op) {
  if constexpr (isArg)
    op.removeArgAttrsAttr();
  else
    op.removeResAttrsAttr();
}

/// Update the given index into an argument or result attribute dictionary.
template <bool isArg>
static void setArgResAttrDict(HWModuleLike op, unsigned numTotalIndices,
                              unsigned index, DictionaryAttr attrs) {
  ArrayAttr allAttrs = getArgResAttrs<isArg>(op);
  if (!allAttrs) {
    if (attrs.empty())
      return;

    // If this attribute is not empty, we need to create a new attribute array.
    SmallVector<Attribute, 8> newAttrs(numTotalIndices,
                                       DictionaryAttr::get(op->getContext()));
    newAttrs[index] = attrs;
    setArgResAttrs<isArg>(op, ArrayAttr::get(op->getContext(), newAttrs));
    return;
  }
  // Check to see if the attribute is different from what we already have.
  if (allAttrs[index] == attrs)
    return;

  // If it is, check to see if the attribute array would now contain only empty
  // dictionaries.
  ArrayRef<Attribute> rawAttrArray = allAttrs.getValue();
  if (attrs.empty() &&
      llvm::all_of(rawAttrArray.take_front(index), isEmptyAttrDict) &&
      llvm::all_of(rawAttrArray.drop_front(index + 1), isEmptyAttrDict))
    return removeArgResAttrs<isArg>(op);

  // Otherwise, create a new attribute array with the updated dictionary.
  SmallVector<Attribute, 8> newAttrs(rawAttrArray.begin(), rawAttrArray.end());
  newAttrs[index] = attrs;
  setArgResAttrs<isArg>(op, ArrayAttr::get(op->getContext(), newAttrs));
}



DictionaryAttr HWModuleLike_impl::getArgAttrDict(HWModuleLike op,
                                                       unsigned index) {
  ArrayAttr attrs = op.getArgAttrsAttr();
  DictionaryAttr argAttrs =
      attrs ? llvm::cast<DictionaryAttr>(attrs[index]) : DictionaryAttr();
  return argAttrs;
}

DictionaryAttr
HWModuleLike_impl::getResultAttrDict(HWModuleLike op,
                                           unsigned index) {
  ArrayAttr attrs = op.getResAttrsAttr();
  DictionaryAttr resAttrs =
      attrs ? llvm::cast<DictionaryAttr>(attrs[index]) : DictionaryAttr();
  return resAttrs;
}

ArrayRef<NamedAttribute>
HWModuleLike_impl::getArgAttrs(HWModuleLike op, unsigned index) {
  auto argDict = getArgAttrDict(op, index);
  return argDict ? argDict.getValue() : std::nullopt;
}

ArrayRef<NamedAttribute>
HWModuleLike_impl::getResultAttrs(HWModuleLike op,
                                        unsigned index) {
  auto resultDict = getResultAttrDict(op, index);
  return resultDict ? resultDict.getValue() : std::nullopt;
}


void HWModuleLike_impl::setArgAttrs(HWModuleLike op,
                                          unsigned index,
                                          ArrayRef<NamedAttribute> attributes) {
  assert(index < op.getNumArguments_HWML() && "invalid argument number");
  return setArgResAttrDict</*isArg=*/true>(
      op, op.getNumArguments_HWML(), index,
      DictionaryAttr::get(op->getContext(), attributes));
}

void HWModuleLike_impl::setArgAttrs(HWModuleLike op,
                                          unsigned index,
                                          DictionaryAttr attributes) {
  return setArgResAttrDict</*isArg=*/true>(
      op, op.getNumArguments_HWML(), index,
      attributes ? attributes : DictionaryAttr::get(op->getContext()));
}

void HWModuleLike_impl::setResultAttrs(
    HWModuleLike op, unsigned index,
    ArrayRef<NamedAttribute> attributes) {
  assert(index < op.getNumResults_HWML() && "invalid result number");
  return setArgResAttrDict</*isArg=*/false>(
      op, op.getNumResults_HWML(), index,
      DictionaryAttr::get(op->getContext(), attributes));
}

void HWModuleLike_impl::setResultAttrs(HWModuleLike op,
                                             unsigned index,
                                             DictionaryAttr attributes) {
  assert(index < op.getNumResults_HWML() && "invalid result number");
  return setArgResAttrDict</*isArg=*/false>(
      op, op.getNumResults_HWML(), index,
      attributes ? attributes : DictionaryAttr::get(op->getContext()));
}

#include "circt/Dialect/HW/HWOpInterfaces.cpp.inc"
