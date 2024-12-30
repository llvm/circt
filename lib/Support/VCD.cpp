//===- VCD.cpp - VCD parser/printer implementation -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/VCD.h"

using namespace circt;
using namespace vcd;

// VCDToken implementation
VCDToken::VCDToken(Kind kind, StringRef spelling) 
  : kind(kind), spelling(spelling) {}

StringRef VCDToken::getSpelling() const { return spelling; }
VCDToken::Kind VCDToken::getKind() const { return kind; }
bool VCDToken::is(Kind K) const { return kind == K; }
bool VCDToken::isCommand() const { return spelling.starts_with("$"); }

SMLoc VCDToken::getLoc() const { 
  return SMLoc::getFromPointer(spelling.data()); 
}

SMLoc VCDToken::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange VCDToken::getLocRange() const { 
  return SMRange(getLoc(), getEndLoc()); 
}

// VCDLexer implementation 
VCDLexer::VCDLexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context)
    : sourceMgr(sourceMgr),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)),
      curBuffer(sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()),
      curPtr(curBuffer.begin()),
      curToken(lexTokenImpl()) {}

// Move all the lexer implementation methods from the header...

// VCDParser implementation
VCDParser::VCDParser(mlir::MLIRContext *context, VCDLexer &lexer)
    : context(context), lexer(lexer) {}

// Move all the parser implementation methods from the header...

// SignalMapping implementation
SignalMapping::SignalMapping(mlir::ModuleOp moduleOp, const VCDFile &file,
                            igraph::InstanceGraph &instanceGraph,
                            igraph::InstancePathCache &instancePathCache,
                            ArrayRef<StringAttr> pathToTop, 
                            VCDFile::Scope *topScope,
                            FlatSymbolRefAttr topModuleName)
    : moduleOp(moduleOp), topScope(topScope), topModuleName(topModuleName),
      file(file), instanceGraph(instanceGraph),
      instancePathCache(instancePathCache) {
  // Move constructor implementation from header...
}

// Move remaining SignalMapping implementation methods...

std::unique_ptr<VCDFile> importVCDFile(llvm::SourceMgr &sourceMgr,
                                       MLIRContext *context) {
  VCDLexer lexer(sourceMgr, context);
  VCDParser parser(context, lexer);
  std::unique_ptr<VCDFile> file;
  if (parser.parseVCDFile(file))
    return nullptr;
  return std::move(file);
}
