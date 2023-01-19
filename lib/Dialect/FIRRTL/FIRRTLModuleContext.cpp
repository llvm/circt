//===- FIRRTLModuleContext.cpp - Module context base ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The implementation of class `FIRRTLModuleContext`.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLModuleContext.h"
#include "llvm/Support/SMLoc.h"

using namespace circt;
using namespace firrtl;

using llvm::SMLoc;

FIRRTLModuleContext::FIRRTLModuleContext(std::string moduleTarget)
    : moduleTarget(std::move(moduleTarget)) {}

Value &FIRRTLModuleContext::getCachedSubaccess(Value value, unsigned index) {
  auto &result = subaccessCache[{value, index}];
  if (!result) {
    // The outer most block won't be in the map.
    auto it = scopeMap.find(value.getParentBlock());
    if (it != scopeMap.end())
      it->second->scopedSubaccesses.emplace_back(result, index);
  }
  return result;
}

ParseResult
FIRRTLModuleContext::addSymbolEntry(StringRef name, SymbolValueEntry entry,
                                    SMLoc loc, bool insertNameIntoGlobalScope) {
  // Do a lookup by trying to do an insertion.  Do so in a way that we can tell
  // if we hit a missing element (SMLoc is null).
  auto entryIt =
      symbolTable.try_emplace(name, SMLoc(), SymbolValueEntry()).first;
  if (entryIt->second.first.isValid()) {
    emitError(loc, "redefinition of name '" + name + "'")
            .attachNote(translateLocation(entryIt->second.first))
        << "previous definition here";
    return failure();
  }

  // If we didn't have a hit, then record the location, and remember that this
  // was new to this scope.
  entryIt->second = {loc, entry};
  if (currentScope && !insertNameIntoGlobalScope)
    currentScope->scopedDecls.push_back(&*entryIt);

  return success();
}

ParseResult
FIRRTLModuleContext::addSymbolEntry(StringRef name, Value value, SMLoc loc,
                                    bool insertNameIntoGlobalScope) {
  return addSymbolEntry(name, SymbolValueEntry(value), loc,
                        insertNameIntoGlobalScope);
}

ParseResult FIRRTLModuleContext::resolveSymbolEntry(Value &result,
                                                    SymbolValueEntry &entry,
                                                    SMLoc loc, bool fatal) {
  if (!entry.is<Value>()) {
    if (fatal)
      emitError(loc, "bundle value should only be used from subfield");
    return failure();
  }
  result = entry.get<Value>();
  return success();
}

ParseResult FIRRTLModuleContext::resolveSymbolEntry(Value &result,
                                                    SymbolValueEntry &entry,
                                                    StringRef fieldName,
                                                    SMLoc loc) {
  if (!entry.is<UnbundledID>()) {
    emitError(loc, "value should not be used from subfield");
    return failure();
  }

  auto fieldAttr = StringAttr::get(getContext(), fieldName);

  unsigned unbundledId = entry.get<UnbundledID>() - 1;
  assert(unbundledId < unbundledValues.size());
  UnbundledValueEntry &ubEntry = unbundledValues[unbundledId];
  for (auto elt : ubEntry) {
    if (elt.first == fieldAttr) {
      result = elt.second;
      break;
    }
  }
  if (!result) {
    emitError(loc, "use of invalid field name '")
        << fieldName << "' on bundle value";
    return failure();
  }

  return success();
}

ParseResult FIRRTLModuleContext::lookupSymbolEntry(SymbolValueEntry &result,
                                                   StringRef name, SMLoc loc) {
  auto &entry = symbolTable[name];
  if (!entry.first.isValid())
    return emitError(loc, "use of unknown declaration '" + name + "'");
  result = entry.second;
  assert(result && "name in symbol table without definition");
  return success();
}

auto FIRRTLModuleContext::getUnbundledEntry(unsigned index)
    -> UnbundledValueEntry & {
  assert(index < unbundledValues.size());
  return unbundledValues[index];
}

FIRRTLModuleContext::ContextScope::ContextScope(
    FIRRTLModuleContext &moduleContext, Block *block)
    : moduleContext(moduleContext), block(block),
      previousScope(moduleContext.currentScope) {
  moduleContext.currentScope = this;
  moduleContext.scopeMap[block] = this;
}

FIRRTLModuleContext::ContextScope::~ContextScope() {
  // Mark all entries in this scope as being invalid.  We track validity
  // through the SMLoc field instead of deleting entries.
  for (auto *entryPtr : scopedDecls)
    entryPtr->second.first = SMLoc();
  // Erase the scoped subacceses from the cache. If the block is deleted we
  // could resuse the memory, although the chances are quite small.
  for (auto subaccess : scopedSubaccesses)
    moduleContext.subaccessCache.erase(subaccess);
  // Erase this context from the map.
  moduleContext.scopeMap.erase(block);
  // Reset to the previous scope.
  moduleContext.currentScope = previousScope;
}
