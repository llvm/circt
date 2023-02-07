//===- FIRRTLFieldSource.cpp - Field Source Analysis ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a basic points-to like analysis.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLFieldSource.h"

using namespace circt;
using namespace firrtl;

FieldSource::FieldSource(Operation *operation) {
  FModuleOp mod = cast<FModuleOp>(operation);
  for (auto port : mod.getBodyBlock()->getArguments())
    if (auto ft = dyn_cast<FIRRTLBaseType>(port.getType()))
      if (!ft.isGround())
        makeNodeForValue(port, port, {});
  for (auto &op : *mod.getBodyBlock())
    visitOp(&op);
}

void FieldSource::visitOp(Operation *op) {
  if (auto sf = dyn_cast<SubfieldOp>(op))
    return visitSubfield(sf);
  if (auto si = dyn_cast<SubindexOp>(op))
    return visitSubindex(si);
  if (auto sa = dyn_cast<SubaccessOp>(op))
    return visitSubaccess(sa);
  if (isa<WireOp, NodeOp, RegOp, RegResetOp, BitCastOp>(op))
    return makeNodeForValue(op->getResult(0), op->getResult(0), {});
  if (auto mem = dyn_cast<MemOp>(op))
    return visitMem(mem);
  if (auto inst = dyn_cast<InstanceOp>(op))
    return visitInst(inst);
  // recurse in to regions
  for (auto &r : op->getRegions())
    for (auto &b : r.getBlocks())
      for (auto &op : b)
        visitOp(&op);
}

void FieldSource::visitSubfield(SubfieldOp sf) {
  auto value = sf.getInput();
  const auto* node = nodeForValue(value);
  assert(node && "node should be in the map");
  auto sv = node->path;
  sv.push_back(sf.getFieldIndex());
  makeNodeForValue(sf.getResult(), node->src, sv);
}

void FieldSource::visitSubindex(SubindexOp si) {
  auto value = si.getInput();
  const auto* node = nodeForValue(value);
  assert(node && "node should be in the map");
  auto sv = node->path;
  sv.push_back(si.getIndex());
  makeNodeForValue(si.getResult(), node->src, sv);
}

void FieldSource::visitSubaccess(SubaccessOp sa) {
  auto value = sa.getInput();
  const auto* node = nodeForValue(value);
  assert(node && "node should be in the map");
  auto sv = node->path;
  sv.push_back(-1);
  makeNodeForValue(sa.getResult(), node->src, sv);
}

void FieldSource::visitMem(MemOp mem) {
  for (auto r : mem.getResults())
    makeNodeForValue(r, r, {});
}

void FieldSource::visitInst(InstanceOp inst) {
  for (auto r : inst.getResults())
    makeNodeForValue(r, r, {});
}

const FieldSource::PathNode *FieldSource::nodeForValue(Value v) const {
  auto ii = paths.find(v);
  if (ii == paths.end())
    return nullptr;
  return &ii->second;
}

void FieldSource::makeNodeForValue(Value dst, Value src,
                                   ArrayRef<int64_t> path) {
  auto ii = paths.try_emplace(dst, src, path);
  assert(ii.second && "Double insert into the map");
}
