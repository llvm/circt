//===- FIRParser.h - .fir to FIRRTL dialect parser --------------*- C++ -*-===//
//
// Defines the interface to the .fir file parser.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRPARSER_H
#define CIRCT_DIALECT_FIRPARSER_H

namespace llvm {
class SourceMgr;
}

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace circt {

struct FIRParserOptions {
  /// If this is set to true, the @info locators are ignored, and the locations
  /// are set to the location in the .fir file.
  bool ignoreInfoLocators = false;
};

mlir::OwningModuleRef parseFIRFile(llvm::SourceMgr &sourceMgr,
                                   mlir::MLIRContext *context,
                                   FIRParserOptions options = {});
void registerFIRParserTranslation();

} // namespace circt

#endif // CIRCT_DIALECT_FIRPARSER_H
