//===- RTGTypes.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGTypes.h"
#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace rtg;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTG/IR/RTGTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TargetType
//===----------------------------------------------------------------------===/

LogicalResult TargetType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<StringAttr> entryNames,
                                 ArrayRef<Type> entryTypes) {
  if (entryNames.size() != entryTypes.size())
    return emitError() << "must have the same number of names and types";

  llvm::SmallDenseSet<StringAttr> entryNameSet;
  LogicalResult result = success();
  entryNameSet.reserve(entryNames.size());
  for (auto name : entryNames)
    if (!entryNameSet.insert(name).second) {
      result = failure();
      emitError() << "duplicate entry name '" << name.getValue()
                  << "' in rtg.target type";
    }
  return result;
}

Type TargetType::parse(AsmParser &p) {
  llvm::StringSet<> nameSet;
  SmallVector<Type> types;
  SmallVector<StringAttr> names;
  bool hasDuplicateName = false;

  auto parseResult = p.parseCommaSeparatedList(
      mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
        std::string name;
        Type type;

        auto fieldLoc = p.getCurrentLocation();
        if (p.parseKeywordOrString(&name) || p.parseColon() ||
            p.parseType(type))
          return failure();

        if (!nameSet.insert(name).second) {
          p.emitError(fieldLoc, "duplicate entry name \'" + name + "\'");
          // Continue parsing to print all duplicates, but make sure to error
          // eventually
          hasDuplicateName = true;
        }

        types.push_back(type);
        names.push_back(StringAttr::get(p.getContext(), name));
        return success();
      });

  if (hasDuplicateName)
    return Type();

  if (failed(parseResult))
    return Type();

  return get(p.getContext(), names, types);
}

void TargetType::print(AsmPrinter &p) const {
  p << '<';
  llvm::interleaveComma(llvm::zip(getEntryNames(), getEntryTypes()), p,
                        [&](auto entry) {
                          auto [name, type] = entry;
                          p.printKeywordOrString(name.getValue());
                          p << ": " << type;
                        });
  p << ">";
}

//===----------------------------------------------------------------------===//
// DictType
//===----------------------------------------------------------------------===/

LogicalResult DictType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<StringAttr> entryNames,
                               ArrayRef<Type> entryTypes) {
  if (entryNames.size() != entryTypes.size())
    return emitError() << "must have the same number of names and types";

  llvm::SmallDenseSet<StringAttr> entryNameSet;
  LogicalResult result = success();
  entryNameSet.reserve(entryNames.size());
  for (auto name : entryNames)
    if (!entryNameSet.insert(name).second) {
      result = failure();
      emitError() << "duplicate entry name '" << name.getValue()
                  << "' in rtg.dict type";
    }
  return result;
}

Type DictType::parse(AsmParser &p) {
  llvm::StringSet<> nameSet;
  SmallVector<Type> types;
  SmallVector<StringAttr> names;
  bool hasDuplicateName = false;

  auto parseResult = p.parseCommaSeparatedList(
      mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
        std::string name;
        Type type;

        auto fieldLoc = p.getCurrentLocation();
        if (p.parseKeywordOrString(&name) || p.parseColon() ||
            p.parseType(type))
          return failure();

        if (!nameSet.insert(name).second) {
          p.emitError(fieldLoc, "duplicate entry name \'" + name + "\'");
          // Continue parsing to print all duplicates, but make sure to error
          // eventually
          hasDuplicateName = true;
        }

        types.push_back(type);
        names.push_back(StringAttr::get(p.getContext(), name));
        return success();
      });

  if (hasDuplicateName)
    return Type();

  if (failed(parseResult))
    return Type();

  return get(p.getContext(), names, types);
}

void DictType::print(AsmPrinter &p) const {
  p << '<';
  llvm::interleaveComma(llvm::zip(getEntryNames(), getEntryTypes()), p,
                        [&](auto entry) {
                          auto [name, type] = entry;
                          p.printKeywordOrString(name.getValue());
                          p << ": " << type;
                        });
  p << ">";
}

Type DictType::getTypeOfEntry(StringAttr entryName) {
  for (auto [name, type] : llvm::zip(getEntryNames(), getEntryTypes()))
    if (name == entryName)
      return type;

  return Type();
}

void circt::rtg::RTGDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/RTG/IR/RTGTypes.cpp.inc"
      >();
}
