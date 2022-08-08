//===- TclEmitter.h - Emit Quartus-flavored, Stratix-10 targeted TCL ----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_TCLEMITTERS_QUARTUS_TCLEMITTER_H
#define CIRCT_DIALECT_MSFT_TCLEMITTERS_QUARTUS_TCLEMITTER_H

#include "circt/Dialect/MSFT/TclEmitter.h"

namespace circt {
namespace msft {
class MSFTModuleOp;

struct QuartusTclOutputState : public TclOutputState {
  using TclOutputState::TclOutputState;

  void emit(PhysLocationAttr) override;
  LogicalResult emitLocationAssignment(DynInstDataOpInterface refOp,
                                       PhysLocationAttr,
                                       Optional<StringRef> subpath) override;
  LogicalResult emit(PDPhysRegionOp region) override;
  LogicalResult emit(PDPhysLocationOp loc) override;
  LogicalResult emit(PDRegPhysLocationOp) override;
  LogicalResult emit(DynamicInstanceVerbatimAttrOp attr) override;
};

class QuartusTclEmitter : public TclEmitter {
public:
  using TclEmitter::TclEmitter;
  std::unique_ptr<TclOutputState>
  newOutputState(llvm::raw_ostream &os) override {
    return std::make_unique<QuartusTclOutputState>(*this, os);
  }
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_TCLEMITTERS_QUARTUS_TCLEMITTER_H
