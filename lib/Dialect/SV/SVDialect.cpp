//===- SVDialect.cpp - Implement the SV dialect ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SV dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ManagedStatic.h"

using namespace circt;
using namespace circt::sv;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void SVDialect::initialize() {
  // Register types.
  registerTypes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/SV/SV.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Name conflict resolution
//===----------------------------------------------------------------------===//

/// Return a StringSet that contains all of the reserved names (e.g. Verilog
/// keywords) that we need to avoid for fear of name conflicts.
struct ReservedWordsCreator {
  static void *call() {
    auto set = std::make_unique<StringSet<>>();
    static const char *const reservedWords[] = {
#include "ReservedWords.def"
    };
    for (auto *word : reservedWords)
      set->insert(word);
    return set.release();
  }
};

/// A StringSet that contains all of the reserved names (e.g., Verilog and VHDL
/// keywords) that we need to avoid to prevent naming conflicts.
static llvm::ManagedStatic<StringSet<>, ReservedWordsCreator> reservedWords;

/// Given string \p origName, generate a new name if it conflicts with any
/// keyword or any other name in the set \p recordNames. Use the int \p
/// nextGeneratedNameID as a counter for suffix. Update the \p recordNames with
/// the generated name and return the StringRef.
StringRef circt::sv::resolveKeywordConflict(StringRef origName,
                                            llvm::StringSet<> &recordNames,
                                            size_t &nextGeneratedNameID) {
  auto name = origName;
  // Get the list of reserved words we need to avoid.  We could prepopulate this
  // into the used words cache, but it is large and immutable, so we just query
  // it when needed.
  SmallVector<char, 16> nameBuffer(name.begin(), name.end());
  nameBuffer.push_back('_');
  auto baseSize = nameBuffer.size();

  while (1) {
    // Loop until we get a name that is not a keyword and is unique.
    if (!reservedWords->count(name)) {
      auto itAndInserted = recordNames.insert(name);
      if (itAndInserted.second)
        return itAndInserted.first->getKey();
    }
    // We need to auto-unique it.
    auto suffix = llvm::utostr(nextGeneratedNameID++);
    nameBuffer.append(suffix.begin(), suffix.end());
    name = StringRef(nameBuffer.data(), nameBuffer.size());

    // Chop off the suffix and try again until we get a unique name..
    nameBuffer.resize(baseSize);
  }
}

static bool isValidVerilogCharacterFirst(char ch) {
  return llvm::isAlpha(ch) || ch == '_';
}

static bool isValidVerilogCharacter(char ch) {
  return isValidVerilogCharacterFirst(ch) || llvm::isDigit(ch);
}

/// Legalize the specified name for use in SV output. Auto-uniquifies the name
/// through \c resolveKeywordConflict if required. If the name is empty, a
/// unique temp name is created.
StringRef circt::sv::legalizeName(StringRef name,
                                  llvm::StringSet<> &recordNames,
                                  size_t &nextGeneratedNameID) {
  if (name.empty())
    name = "_T";

  // Check to see if the name consists of all-valid identifiers.  If not, we
  // need to escape them.
  for (char ch : name) {
    if (isValidVerilogCharacter(ch))
      continue;

    // Otherwise, we need to escape it.
    SmallString<16> tmpName;
    for (char ch : name) {
      if (isValidVerilogCharacter(ch))
        tmpName += ch;
      else if (ch == ' ' || ch == '.')
        tmpName += '_';
      else {
        tmpName += llvm::utohexstr((unsigned char)ch);
      }
    }
    return legalizeName(tmpName, recordNames, nextGeneratedNameID);
  }

  // Check to see if this name is valid.  The first character cannot be a
  // number or some other weird thing.  If it is, start with an underscore.
  if (!isValidVerilogCharacterFirst(name.front())) {
    SmallString<16> tmpName("_");
    tmpName += name;
    return legalizeName(tmpName, recordNames, nextGeneratedNameID);
  }

  // Make sure the new valid name does not conflict with any existing names.
  return resolveKeywordConflict(name, recordNames, nextGeneratedNameID);
}

/// Check if a name is valid for use in SV output by only containing characters
/// allowed in SV identifiers.
///
/// Call \c legalizeName() to obtain a legalized version of the name.
bool circt::sv::isNameValid(StringRef name) {
  if (name.empty())
    return false;
  if (!isValidVerilogCharacterFirst(name.front()))
    return false;
  for (char ch : name) {
    if (!isValidVerilogCharacter(ch))
      return false;
  }
  return reservedWords->count(name) == 0;
}
