//===- EmissionPrinter.h - Provides printing utilites to ExportSystemC ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This declares the EmissionPrinter class to provide printing utilities to
// ExportSystemC.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_TARGET_EXPORTSYSTEMC_EMISSIONPRINTER_H
#define CIRCT_TARGET_EXPORTSYSTEMC_EMISSIONPRINTER_H

#include "EmissionPattern.h"
#include "mlir/Support/IndentedOstream.h"

namespace circt {
namespace ExportSystemC {

/// This is intended to be the driving class for all pattern-based IR emission.
class EmissionPrinter {
public:
  EmissionPrinter(mlir::raw_indented_ostream &os,
                  const FrozenOpEmissionPatternSet &opPatterns,
                  const FrozenTypeEmissionPatternSet &typePatterns,
                  Location loc)
      : opPatterns(opPatterns), typePatterns(typePatterns), os(os),
        emissionFailed(false), currentLoc(loc) {}

  EmissionPrinter(mlir::raw_indented_ostream &os,
                  OpEmissionPatternSet &opPatterns,
                  TypeEmissionPatternSet &typePatterns, Location loc)
      : opPatterns(std::move(opPatterns)),
        typePatterns(std::move(typePatterns)), os(os), emissionFailed(false),
        currentLoc(loc) {}

  /// Emit the given operation as a statement to the ostream associated with
  /// this printer according to the emission patterns registered. An operation
  /// might also emit multiple statements, or nothing in case it can only be
  /// emitted as an expression. If multiple emission patterns match, the first
  /// one in the first one in the pattern set is chosen. If no pattern matches,
  /// a remark is left in the output and an error is added to stderr.
  /// Additionally, the exit-code to be obtained by the 'exitCode()'
  /// member-function is set to 'failure'.
  void emitOp(Operation *op);

  /// Emit the expression represented by the given value to the ostream
  /// associated with this printer according to the emission patterns
  /// registered. This will emit exactly one expression and does not emit any
  /// statements. If multiple emission patterns match, the first one in the
  /// first one in the pattern set is chosen. If no pattern matches, a remark is
  /// left in the output and an error is added to stderr. Additionally, the
  /// exit-code to be obtained by the 'exitCode()' member-function is set to
  /// 'failure'.
  InlineEmitter getInlinable(Value value);

  /// Emit the given type to the ostream associated with this printer according
  /// to the emission patterns registered. If multiple emission patterns match,
  /// the first one in the pattern set is chosen. If no pattern matches, a
  /// remark is left in the output and an error is added to stderr.
  /// Additionally, the exit-code to be obtained by the 'exitCode()'
  /// member-function is set to 'failure'.
  void emitType(Type type);

  /// Emit the given region to the ostream associated with this printer. Only
  /// regions with a single basic block are allowed. Prints the operations
  /// inside according to 'emitOp()' indented one level deeper and encloses the
  /// region in curly-braces.
  void emitRegion(Region &region);

  /// Emit the given region to the ostream associated with this printer. Only
  /// regions with a single basic block are allowed. Prints the operations
  /// inside according to 'emitOp()'. The enclosing delimiters and level of
  /// indentation is determined by the passed scope.
  void emitRegion(Region &region,
                  mlir::raw_indented_ostream::DelimitedScope &scope);

  EmissionPrinter &operator<<(StringRef str);
  EmissionPrinter &operator<<(int64_t num);

  mlir::raw_indented_ostream &getOstream() const { return os; }

  /// Returns whether everything was printed successfully or some error occurred
  /// (e.g., there was an operation or type for which no emission pattern was
  /// valid).
  LogicalResult exitState() const { return failure(emissionFailed); }

private:
  FrozenOpEmissionPatternSet opPatterns;
  FrozenTypeEmissionPatternSet typePatterns;
  mlir::raw_indented_ostream &os;
  bool emissionFailed;
  Location currentLoc;
};

} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_EMISSIONPRINTER_H
