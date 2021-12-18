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
public:
  AnnoPathValue(Operation *op = nullptr, unsigned fieldIdx = 0,
                ArrayRef<InstanceOp> path = {}) : op(op), portNum(~0ULL), fieldIdx(fieldIdx), instances(path.begin(), path.end()) {}
  AnnoPathValue(FModuleLike mod, unsigned fieldIdx = 0, size_t portNum = ~0ULL,
                ArrayRef<InstanceOp> path = {})
      : op(mod), portNum(portNum), fieldIdx(fieldIdx),
        instances(path.begin(), path.end()) {}

  bool isLocal() const { return instances.empty(); }
  bool isPort() const { return op && portNum != ~0ULL; }
  bool isInstance() const { return op && isa<InstanceOp>(op); }
  
  FIRRTLType getType() const {
    FIRRTLType t;
    if (isPort()) {
      if (auto mod = dyn_cast<FModuleLike>(op))
        t = mod.getPortType(portNum);
      else if (auto inst = dyn_cast<InstanceOp>(op))
        t = inst.getResult(portNum).getType().cast<FIRRTLType>();
      else if (auto mem = dyn_cast<MemOp>(op))
        t = mem.getResult(portNum).getType().cast<FIRRTLType>();
    } else
      t = op->getResultTypes()[0].cast<FIRRTLType>();
    if (!t)
      return t;
    return t.getSubTypeByFieldID(fieldIdx);
  }
  ArrayRef<InstanceOp> getPath() const { return instances; }
  unsigned getField() const { return fieldIdx; }
  size_t getPort() const { return portNum; }
  FModuleOp getModule() const {
    if (auto mod = dyn_cast<FModuleOp>(op))
     return mod;
    return op->getParentOfType<FModuleOp>();
    
  }
  Operation *getOp() const { return op; }
  ArrayRef<InstanceOp> getInstances() const { return instances; }
  StringAttr getName() const {
    if (isPort()) {
      if (auto mod = dyn_cast<FModuleLike>(op))
        return StringAttr::get(getContext(), mod.getPortName(portNum));
      else if (auto inst = dyn_cast<InstanceOp>(op))
        return inst.getPortName(portNum);
      else if (auto mem = dyn_cast<MemOp>(op))
        return mem.getPortName(portNum);
    }
    if (auto mod = dyn_cast<FModuleLike>(op))
      return StringAttr::get(getContext(), mod.moduleName());
      return op->getAttrOfType<StringAttr>("name");
  }

  void setPort(size_t port) { portNum = port; }
  void setField(unsigned field)  { fieldIdx = field; }
  void setPath(ArrayRef<InstanceOp> path) {
    instances.clear();
    instances.append(path.begin(), path.end());
  }

  explicit operator bool() const { return op != nullptr; }

  template <typename... T>
  bool isOpOfType() const {
    if (!op || isPort())
      return false;
    return isa<T...>(op);
  }

  mlir::MLIRContext* getContext() const { return op->getContext(); }

private:
  Operation *op;
  size_t portNum;
  unsigned fieldIdx = 0;
  SmallVector<InstanceOp, 4> instances;
};

/// State threaded through functions for resolving and applying annotations.
class AnnoApplyState {
public:
  AnnoApplyState(size_t id, CircuitOp circuit, SymbolTable& symTbl)
  :id(id), circuit(circuit), symTbl(symTbl) {}

  LogicalResult applyAnnoToTarget(StringRef, NamedAttrList);
  LogicalResult applyAnnoToOp(Operation*, NamedAttrList);
  LogicalResult setDontTouch(StringRef target);

  IntegerAttr newID();

  MLIRContext* getContext() { return circuit.getContext(); }
  CircuitOp getCircuit() const { return circuit;}
  SymbolTable& getSymbolTable() const { return symTbl; }
  Location getLoc() { return circuit.getLoc(); }

private:
  size_t id;
  CircuitOp circuit;
  SymbolTable &symTbl;
};

//===----------------------------------------------------------------------===//
// Annotation lowering utilities
//===----------------------------------------------------------------------===//

/// Mutably update a prototype Annotation (stored as a `NamedAttrList`) with
/// subfield/subindex information from a Target string.  Subfield/subindex
/// information will be placed in the key "target" at the back of the
/// Annotation.  If no subfield/subindex information, the Annotation is
/// unmodified.  Return the split input target as a base target (include a
/// reference if one exists) and an optional array containing subfield/subindex
/// tokens.
std::pair<StringRef, llvm::Optional<ArrayAttr>>
splitAndAppendTarget(NamedAttrList &annotation, StringRef target,
                     MLIRContext *context);

/// Split out non-local paths.  This will return a set of target strings for
/// each named entity along the path.
/// c|c:ai/Am:bi/Bm>d.agg[3] ->
/// c|c>ai, c|Am>bi, c|Bm>d.agg[2]
SmallVector<std::tuple<std::string, StringRef, StringRef>>
expandNonLocal(StringRef target);

/// Make an anchor for a non-local annotation.  Use the expanded path to build
/// the module and name list in the anchor.
FlatSymbolRefAttr buildNLA(AnnoPathValue target, AnnoApplyState state);

/// Split a target into a base target (including a reference if one exists) and
/// an optional array of subfield/subindex tokens.
std::pair<StringRef, llvm::Optional<ArrayAttr>>
splitTarget(StringRef target, MLIRContext *context);

StringRef getAnnoClass(DictionaryAttr anno);

SmallString<32> canonicalizeTarget(StringRef target);
StringAttr canonicalizeTarget(StringAttr target);

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

LogicalResult applyOMIR(AnnoPathValue target, DictionaryAttr anno,
                         AnnoApplyState state);

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_ANNOTATION_LOWERING_H
