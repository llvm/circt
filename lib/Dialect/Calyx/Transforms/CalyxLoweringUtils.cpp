//===- CalyxLoweringUtils.cpp - Calyx lowering utility methods --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various lowering utility methods converting to and from Calyx programs.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"

#include <variant>

namespace circt {
namespace calyx {

WalkResult
getCiderSourceLocationMetadata(calyx::ComponentOp component,
                               SmallVectorImpl<Attribute> &sourceLocations) {
  Builder builder(component->getContext());
  return component.getControlOp().walk([&](Operation *op) {
    if (!calyx::isControlLeafNode(op))
      return WalkResult::advance();

    std::string sourceLocation;
    llvm::raw_string_ostream os(sourceLocation);
    op->getLoc()->print(os);
    int64_t position = sourceLocations.size();
    sourceLocations.push_back(
        StringAttr::get(op->getContext(), sourceLocation));

    op->setAttr("pos", builder.getI64IntegerAttr(position));
    return WalkResult::advance();
  });
}

bool matchConstantOp(Operation *op, APInt &value) {
  return mlir::detail::constant_int_op_binder(&value).match(op);
}

bool singleLoadFromMemory(Value memoryReference) {
  return llvm::count_if(memoryReference.getUses(), [](OpOperand &user) {
           return isa<mlir::memref::LoadOp>(user.getOwner());
         }) <= 1;
}

bool noStoresToMemory(Value memoryReference) {
  return llvm::none_of(memoryReference.getUses(), [](OpOperand &user) {
    return isa<mlir::memref::StoreOp>(user.getOwner());
  });
}

Value getComponentOutput(calyx::ComponentOp compOp, unsigned outPortIdx) {
  size_t index = compOp.getInputPortInfo().size() + outPortIdx;
  assert(index < compOp.getNumArguments() &&
         "Exceeded number of arguments in the Component");
  return compOp.getArgument(index);
}

Type convIndexType(OpBuilder &builder, Type type) {
  if (type.isIndex())
    return builder.getI32Type();
  return type;
}

void buildAssignmentsForRegisterWrite(OpBuilder &builder,
                                      calyx::GroupOp groupOp,
                                      calyx::ComponentOp componentOp,
                                      calyx::RegisterOp &reg,
                                      Value inputValue) {
  mlir::IRRewriter::InsertionGuard guard(builder);
  auto loc = inputValue.getLoc();
  builder.setInsertionPointToEnd(groupOp.getBody());
  builder.create<calyx::AssignOp>(loc, reg.in(), inputValue);
  builder.create<calyx::AssignOp>(
      loc, reg.write_en(), createConstant(loc, builder, componentOp, 1, 1));
  builder.create<calyx::GroupDoneOp>(loc, reg.done());
}

//===----------------------------------------------------------------------===//
// MemoryInterface
//===----------------------------------------------------------------------===//

MemoryInterface::MemoryInterface() {}
MemoryInterface::MemoryInterface(const MemoryPortsImpl &ports) : impl(ports) {}
MemoryInterface::MemoryInterface(calyx::MemoryOp memOp) : impl(memOp) {}

Value MemoryInterface::readData() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->readData();
  }
  return std::get<MemoryPortsImpl>(impl).readData;
}

Value MemoryInterface::done() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->done();
  }
  return std::get<MemoryPortsImpl>(impl).done;
}

Value MemoryInterface::writeData() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->writeData();
  }
  return std::get<MemoryPortsImpl>(impl).writeData;
}

Value MemoryInterface::writeEn() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->writeEn();
  }
  return std::get<MemoryPortsImpl>(impl).writeEn;
}

ValueRange MemoryInterface::addrPorts() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->addrPorts();
  }
  return std::get<MemoryPortsImpl>(impl).addrPorts;
}

//===----------------------------------------------------------------------===//
// LoopInterface
//===----------------------------------------------------------------------===//

LoopInterface::~LoopInterface() = default;

//===----------------------------------------------------------------------===//
// ComponentLoweringStateInterface
//===----------------------------------------------------------------------===//

ComponentLoweringStateInterface::ComponentLoweringStateInterface(
    calyx::ComponentOp component)
    : component(component) {}

ComponentLoweringStateInterface::~ComponentLoweringStateInterface() = default;

calyx::ComponentOp ComponentLoweringStateInterface::getComponentOp() {
  return component;
}

void ComponentLoweringStateInterface::addBlockArgReg(Block *block,
                                                     calyx::RegisterOp reg,
                                                     unsigned idx) {
  assert(blockArgRegs[block].count(idx) == 0);
  assert(idx < block->getArguments().size());
  blockArgRegs[block][idx] = reg;
}

const DenseMap<unsigned, calyx::RegisterOp> &
ComponentLoweringStateInterface::getBlockArgRegs(Block *block) {
  return blockArgRegs[block];
}

void ComponentLoweringStateInterface::addBlockArgGroup(Block *from, Block *to,
                                                       calyx::GroupOp grp) {
  blockArgGroups[from][to].push_back(grp);
}

ArrayRef<calyx::GroupOp>
ComponentLoweringStateInterface::getBlockArgGroups(Block *from, Block *to) {
  return blockArgGroups[from][to];
}

std::string ComponentLoweringStateInterface::getUniqueName(StringRef prefix) {
  std::string prefixStr = prefix.str();
  unsigned idx = prefixIdMap[prefixStr];
  ++prefixIdMap[prefixStr];
  return (prefix + "_" + std::to_string(idx)).str();
}

StringRef ComponentLoweringStateInterface::getUniqueName(Operation *op) {
  auto it = opNames.find(op);
  assert(it != opNames.end() && "A unique name should have been set for op");
  return it->second;
}

void ComponentLoweringStateInterface::setUniqueName(Operation *op,
                                                    StringRef prefix) {
  assert(opNames.find(op) == opNames.end() &&
         "A unique name was already set for op");
  opNames[op] = getUniqueName(prefix);
}

void ComponentLoweringStateInterface::registerEvaluatingGroup(
    Value v, calyx::GroupInterface group) {
  valueGroupAssigns[v] = group;
}

void ComponentLoweringStateInterface::addReturnReg(calyx::RegisterOp reg,
                                                   unsigned idx) {
  assert(returnRegs.count(idx) == 0 &&
         "A register was already registered for this index");
  returnRegs[idx] = reg;
}

calyx::RegisterOp ComponentLoweringStateInterface::getReturnReg(unsigned idx) {
  assert(returnRegs.count(idx) && "No register registered for index!");
  return returnRegs[idx];
}

void ComponentLoweringStateInterface::registerMemoryInterface(
    Value memref, const calyx::MemoryInterface &memoryInterface) {
  assert(memref.getType().isa<MemRefType>());
  assert(memories.find(memref) == memories.end() &&
         "Memory already registered for memref");
  memories[memref] = memoryInterface;
}

calyx::MemoryInterface
ComponentLoweringStateInterface::getMemoryInterface(Value memref) {
  assert(memref.getType().isa<MemRefType>());
  auto it = memories.find(memref);
  assert(it != memories.end() && "No memory registered for memref");
  return it->second;
}

Optional<calyx::MemoryInterface>
ComponentLoweringStateInterface::isInputPortOfMemory(Value v) {
  for (auto &memIf : memories) {
    auto &mem = memIf.getSecond();
    if (mem.writeEn() == v || mem.writeData() == v ||
        llvm::any_of(mem.addrPorts(), [=](Value port) { return port == v; }))
      return {mem};
  }
  return {};
}

void ComponentLoweringStateInterface::setFuncOpResultMapping(
    const DenseMap<unsigned, unsigned> &mapping) {
  funcOpResultMapping = mapping;
}

unsigned ComponentLoweringStateInterface::getFuncOpResultMapping(
    unsigned funcReturnIdx) {
  auto it = funcOpResultMapping.find(funcReturnIdx);
  assert(it != funcOpResultMapping.end() &&
         "No component return port index recorded for the requested function "
         "return index");
  return it->second;
}

//===----------------------------------------------------------------------===//
// ModuleOpConversion
//===----------------------------------------------------------------------===//

ModuleOpConversion::ModuleOpConversion(MLIRContext *context,
                                       StringRef topLevelFunction,
                                       calyx::ProgramOp *programOpOutput)
    : OpRewritePattern<mlir::ModuleOp>(context),
      programOpOutput(programOpOutput), topLevelFunction(topLevelFunction) {
  assert(programOpOutput->getOperation() == nullptr &&
         "this function will set programOpOutput post module conversion");
}

LogicalResult
ModuleOpConversion::matchAndRewrite(mlir::ModuleOp moduleOp,
                                    PatternRewriter &rewriter) const {
  if (!moduleOp.getOps<calyx::ProgramOp>().empty())
    return failure();

  rewriter.updateRootInPlace(moduleOp, [&] {
    // Create ProgramOp
    rewriter.setInsertionPointAfter(moduleOp);
    auto programOp = rewriter.create<calyx::ProgramOp>(
        moduleOp.getLoc(), StringAttr::get(getContext(), topLevelFunction));

    // Inline the module body region
    rewriter.inlineRegionBefore(moduleOp.getBodyRegion(),
                                programOp.getBodyRegion(),
                                programOp.getBodyRegion().end());

    // Inlining the body region also removes ^bb0 from the module body
    // region, so recreate that, before finally inserting the programOp
    auto *moduleBlock = rewriter.createBlock(&moduleOp.getBodyRegion());
    rewriter.setInsertionPointToStart(moduleBlock);
    rewriter.insert(programOp);
    *programOpOutput = programOp;
  });
  return success();
}

} // namespace calyx
} // namespace circt
