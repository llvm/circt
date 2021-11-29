//===- FIRRTLAnnotationLowering.h - Code for  lowering Annos ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helpers for lowering with FIRRTL annotations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_ANNOTATION_LOWERING_H
#define CIRCT_DIALECT_FIRRTL_ANNOTATION_LOWERING_H


namespace circt {
namespace firrtl {

//===----------------------------------------------------------------------===//
// Data to/from the lowerer
//===----------------------------------------------------------------------===//

class AnnoPathValue {
  AnnoPathValue(Operation *op, unsigned fieldIdx = 0, ArrayRef<InstanceOp> path = {});
  AnnoPathValue(FModuleLike *mod, unsigned fieldIdx = 0, size_t portNum = ~0ULL,
                ArrayRef<InstanceOp> path = {});
  FIRRTLType getType() const;
  ArrayRef<InstanceOp> getPath() const;
  bool isLocal() const;
  bool isPort() const;
  bool isInstance() const;
  FModuleOp getModule() const;

  template <typename... T>
  bool isOpOfType() const {
    if (!op || isPort())
      return false;
    return isa<T...>(op);
  }

  private:
    SmallVector<InstanceOp, 4> instances;
    Operation *op;
    size_t portNum;
    unsigned fieldIdx = 0;
};

/// State threaded through functions for resolving and applying annotations.
struct AnnoApplyState {
  AnnoApplyState(SymbolTable &symTbl) : symTbl(symTbl) {}
  CircuitOp circuit;
  SymbolTable &symTbl;
  llvm::function_ref<void(DictionaryAttr)> addToWorklistFn;
  size_t newID() { return ++id; }

private:
  size_t id;
};

//===----------------------------------------------------------------------===//
// Pass Specific Annotation lowering
//===----------------------------------------------------------------------===//

LogicalResult applyModRep(AnnoPathValue target, DictionaryAttr anno,
                          AnnoApplyState state);
LogicalResult applyGCView(AnnoPathValue target, DictionaryAttr anno,
                          AnnoApplyState state);
LogicalResult applyGCSigDriver(AnnoPathValue target, DictionaryAttr anno,
                               AnnoApplyState state);
LogicalResult applyGCDataTap(AnnoPathValue target, DictionaryAttr anno,
                             AnnoApplyState state);
LogicalResult applyGCMemTap(AnnoPathValue target, DictionaryAttr anno,
                            AnnoApplyState state);

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_ANNOTATION_LOWERING_H
