//===- ImportVerilogDebugStream.h - A debug stream wrapper for importverilog
//---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IMPORT_VERILOG_DEBUG_H
#define IMPORT_VERILOG_DEBUG_H

#include "mlir/IR/Location.h"
#include "slang/ast/Compilation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <source_location>
#include <utility>

// If includer defined DEBUG_TYPE, use that; else fall back to our default.
#ifndef IMPORTVERILOG_DEBUG_TYPE
  #ifdef DEBUG_TYPE
    #define IMPORTVERILOG_DEBUG_TYPE DEBUG_TYPE
  #else
    #define IMPORTVERILOG_DEBUG_TYPE "import-verilog"
  #endif
#endif

struct ImportVerilogDebugStream {
  std::optional<mlir::Location> srcLoc;
  std::optional<std::source_location> compLoc;
  llvm::SmallString<128> buffer;
  llvm::raw_svector_ostream os;

  ImportVerilogDebugStream(std::optional<mlir::Location> sl,
                           std::optional<std::source_location> cl)
      : srcLoc(sl), compLoc(cl), os(buffer) {}

  inline ~ImportVerilogDebugStream() {
    DEBUG_WITH_TYPE(IMPORTVERILOG_DEBUG_TYPE, {
      if(compLoc)
        llvm::dbgs() << compLoc->file_name() << ":" << compLoc->line() << " ("
                     << compLoc->function_name() << ") ";
      if (srcLoc) {
        srcLoc->print(llvm::dbgs());
        llvm::dbgs() << ": ";
      }
      llvm::dbgs() << buffer << "\n";
    });
  }

  inline llvm::raw_ostream &stream() { return os; }
  inline const llvm::raw_ostream &stream() const { return os; }
};

// lvalue wrapper
template <typename T>
inline ImportVerilogDebugStream &operator<<(ImportVerilogDebugStream &s,
                                            const T &v) {
  s.stream() << v;
  return s;
}

// rvalue/temporary wrapper
template <typename T>
inline ImportVerilogDebugStream &&operator<<(ImportVerilogDebugStream &&s,
                                             const T &v) {
  s.stream() << v;
  return std::move(s);
}

// support ostream manipulators that look like functions, e.g. write_hex, etc.
inline ImportVerilogDebugStream &
operator<<(ImportVerilogDebugStream &s,
           llvm::raw_ostream &(*manip)(llvm::raw_ostream &)) {
  manip(s.stream());
  return s;
}
inline ImportVerilogDebugStream &&
operator<<(ImportVerilogDebugStream &&s,
           llvm::raw_ostream &(*manip)(llvm::raw_ostream &)) {
  manip(s.stream());
  return std::move(s);
}

/// Helper function to set up a debug stream with reasonable defaults
ImportVerilogDebugStream dbgs(
                              std::optional<mlir::Location> sourceLocation = {},
    std::optional<std::source_location> cl = std::source_location::current());

#endif // IMPORT_VERILOG_DEBUG_H
