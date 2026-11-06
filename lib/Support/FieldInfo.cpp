//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/FieldInfo.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseSet.h"

using namespace circt;

ParseResult circt::parseFieldList(
    AsmParser &p, SmallVectorImpl<FieldInfo> &fields,
    SmallVectorImpl<std::pair<llvm::SMLoc, StringAttr>> &duplicates) {
  llvm::SmallDenseSet<StringAttr> nameSet;
  return p.parseCommaSeparatedList(
      mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
        std::string name;
        Type type;

        auto fieldLoc = p.getCurrentLocation();
        if (p.parseKeywordOrString(&name) || p.parseColon() ||
            p.parseType(type))
          return failure();

        auto nameAttr = StringAttr::get(p.getContext(), name);
        if (!nameSet.insert(nameAttr).second)
          duplicates.push_back({fieldLoc, nameAttr});

        fields.push_back(FieldInfo{nameAttr, type});
        return success();
      });
}

void circt::printFieldList(AsmPrinter &p, ArrayRef<FieldInfo> fields) {
  p << '<';
  llvm::interleaveComma(fields, p, [&](const FieldInfo &field) {
    p.printKeywordOrString(field.name.getValue());
    p << ": " << field.type;
  });
  p << ">";
}
