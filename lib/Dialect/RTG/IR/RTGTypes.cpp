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
// DictType
//===----------------------------------------------------------------------===/

LogicalResult DictType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<DictEntry> entries) {
  StringAttr last;
  for (auto entry : entries) {
    if (entry.name.empty())
      return emitError() << "empty strings not allowed as entry names";

    if (last && entry.name.getValue() <= last.getValue())
      return emitError() << "dictionary must be sorted by names and contain no "
                            "duplicates, first violation at entry '"
                         << entry.name.getValue() << "'";

    last = entry.name;
  }

  return success();
}

Type DictType::parse(AsmParser &p) {
  SmallVector<DictEntry> entries;
  auto loc = p.getCurrentLocation();

  auto parseResult = p.parseCommaSeparatedList(
      mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
        std::string name;
        Type type;
        loc = p.getCurrentLocation();

        if (p.parseKeywordOrString(&name) || p.parseColon() ||
            p.parseType(type))
          return failure();

        DictEntry entry;
        entry.name = StringAttr::get(p.getContext(), name);
        entry.type = type;
        entries.emplace_back(entry);
        return success();
      });

  if (failed(parseResult))
    return Type();

  auto emitError = [&]() { return p.emitError(loc); };

  // Call 'getChecked' here such that we do not have to repeat the verification
  // checks in the parser here, but still get the errors reported at meaningful
  // locations.
  return getChecked(emitError, p.getContext(), entries);
}

void DictType::print(AsmPrinter &p) const {
  p << '<';
  llvm::interleaveComma(getEntries(), p, [&](auto entry) {
    p.printKeywordOrString(entry.name.getValue());
    p << ": " << entry.type;
  });
  p << ">";
}

bool DictType::entryTypesMatch(TypeRange types) const {
  return llvm::equal(getEntries(), types,
                     [](const DictEntry &entry, const Type &type) {
                       return entry.type == type;
                     });
}

void circt::rtg::RTGDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/RTG/IR/RTGTypes.cpp.inc"
      >();
}
