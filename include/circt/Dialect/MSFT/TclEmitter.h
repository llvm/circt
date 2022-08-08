//===- TclEmitter.h - MSFT Tcl Exporters ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Expose the Tcl exporters.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_EXPORTTCL_H
#define CIRCT_DIALECT_MSFT_EXPORTTCL_H

#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace msft {
class MSFTModuleOp;

class TclEmitter;
struct TclOutputState {
  TclOutputState(TclEmitter &emitter, llvm::raw_ostream &os)
      : os(os), emitter(emitter) {}

  virtual ~TclOutputState();

  llvm::raw_ostream &os;
  llvm::raw_ostream &indent() {
    os.indent(2);
    return os;
  };

  TclEmitter &emitter;
  SmallVector<Attribute> symbolRefs;

  virtual void emit(PhysLocationAttr);
  virtual LogicalResult emitLocationAssignment(DynInstDataOpInterface refOp,
                                               PhysLocationAttr,
                                               Optional<StringRef> subpath);

  virtual LogicalResult emit(PDPhysRegionOp region);
  virtual LogicalResult emit(PDPhysLocationOp loc);
  virtual LogicalResult emit(PDRegPhysLocationOp);
  virtual LogicalResult emit(DynamicInstanceVerbatimAttrOp attr);

  void emitPath(hw::GlobalRefOp ref, Optional<StringRef> subpath);
  void emitInnerRefPart(hw::InnerRefAttr innerRef);

  /// Get the GlobalRefOp to which the given operation is pointing. Add it to
  /// the set of used global refs.
  hw::GlobalRefOp getRefOp(DynInstDataOpInterface op);
};

/// Instantiate for all Tcl emissions. We want to cache the symbols and binned
/// ops -- this helper class provides that caching.

class TclEmitter {
public:
  TclEmitter(mlir::ModuleOp topLevel) : topLevel(topLevel), populated(false) {}
  virtual ~TclEmitter() {}

  /// Write out all the relevant tcl commands. Create one 'proc' per module
  /// which takes the parent entity name since we don't assume that the created
  /// module is the top level for the entire design.
  LogicalResult emit(Operation *forMod, StringRef outputFile);

  Operation *getDefinition(FlatSymbolRefAttr);
  const DenseSet<hw::GlobalRefOp> &getRefsUsed() { return refsUsed; }
  void usedRef(hw::GlobalRefOp ref) { refsUsed.insert(ref); }

protected:
  virtual std::unique_ptr<TclOutputState>
  newOutputState(llvm::raw_ostream &os) = 0;

  mlir::ModuleOp topLevel;

  /// Map Module operations to their top-level "instance" names. Map those
  /// "instance" names to the lowered ops which get directly emitted as tcl.
  DenseMap<Operation *,
           llvm::MapVector<StringAttr, SmallVector<DynInstDataOpInterface, 0>>>
      tclOpsForModInstance;
  DenseSet<hw::GlobalRefOp> refsUsed;

  // Populates the symbol cache.
  LogicalResult populate();
  bool populated;
  hw::HWSymbolCache topLevelSymbols;
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_EXPORTTCL_H
