//===- Partition.cpp
//--------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/ValueMapper.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#include <deque>
#include <random>

#define DEBUG_TYPE "hw-partition"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_PARTITION
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {

struct ModuleWeight {
  size_t bitOps = 0;
  size_t bitRegs = 0;
  size_t bitSram = 0;
  size_t numOps = 0;
};

struct NormalizationResult {
  // The new instance
  InstanceOp instop;
  // The values corresponding to original outputs
  SmallVector<Value> outputs;
};

struct PartitionPlan {
  DenseMap<Operation *, size_t> fwdMap;
};

struct PartitionPass : public circt::hw::impl::PartitionBase<PartitionPass> {
public:
  PartitionPass(std::string moduleName)
      : PartitionBase{circt::hw::PartitionOptions{std::move(moduleName)}} {}
  void runOnOperation() override;
private:
  ModuleWeight statModule(HWModuleOp module, DenseMap<HWModuleOp, ModuleWeight> &cache);
  BitVector analyzeCombDep(HWModuleOp module, PortInfo port);
  BitVector analyzeCombDep(HWModuleOp module, Value val);
  NormalizationResult normalizeInst(OpBuilder builder, InstanceOp instop, const BitVector &outputs);
  PortInfo resultToPort(InstanceOp instop, Value outerVal);
  Value outputToResult(InstanceOp instop, Value innerVal);
  SmallVector<Value> findRelatedInput(InstanceOp instop, Value val);
  Value extractFromInst(OpBuilder builder, Value val, InstanceOp instop, IRMapping &extracted);
  DenseMap<HWModuleOp, ModuleWeight> preparePartition(OpBuilder builder, HWModuleOp module);
  PartitionPlan partition(OpBuilder builder, HWModuleOp module);
  void commitPartition(OpBuilder builder, HWModuleOp root, PartitionPlan plan);
  
  // FIXME: move out
  DenseMap<std::pair<HWModuleOp, Value>, BitVector> _comb_deps;
  SymbolCache _sc;
};
} // namespace

static std::string generateModuleName(HWModuleOp moduleOp, size_t i) {
  return moduleOp.getName().str() + "_P" + std::to_string(i);
}

static std::string generateRegPortName(seq::FirRegOp regOp) {
  return std::string{"reg_"} + regOp.getName().str();
}

struct ValueDepInfo {
  Value outputValue;
  SetVector<BlockArgument> args;
  SmallVector<Operation *> ops;
  SetVector<seq::FirRegOp> needCrossRegs;
  DenseSet<Operation *> seenOps; // TODO: Move
};

struct Part {
  ValueDepInfo depInfo; // TODO: rename
  //
  HWModuleOp newMod;
  hw::InstanceOp newInst;
  DenseMap<seq::FirRegOp, size_t /* index */> ownedRegs;
};

class ModuleAnalyzer {
public:
  SmallVector<Part> scheduleParts(HWModuleOp mod) {
    auto outputs = (*mod.getOps<hw::OutputOp>().begin()).getOutputs();

    // Schedule regs. TODO: FIXME
    auto regOps = llvm::to_vector(mod.getOps<seq::FirRegOp>());
    assert(regOps.size() == 3);
    scheduledRegs[regOps[0]] = 2;
    scheduledRegs[regOps[1]] = 1;
    scheduledRegs[regOps[2]] = 0;

    SmallVector<Part> parts;

    for (auto [i, output] : llvm::enumerate(outputs)) {
      auto depInfo = analyzeValueDepInfo(i, output);
      assert(!depInfo.ops.empty());
      parts.emplace_back(Part{std::move(depInfo)});
    }

    return parts;
  }

  ValueDepInfo analyzeValueDepInfo(size_t partIndex, Value value) const {
    ValueDepInfo depInfo;
    depInfo.outputValue = value;
    analyzeValueDepInfoImpl(depInfo, partIndex, value);
    return depInfo;
  }

  size_t getPartIndexForReg(seq::FirRegOp reg) const {
    return scheduledRegs.at(reg);
  }

private:
  DenseMap<Operation *, size_t> scheduledRegs;

  void analyzeValueDepInfoImpl(ValueDepInfo &depInfo, size_t partIndex,
                               Value value) const {
    if (auto op = value.getDefiningOp()) {
      auto [iter, insert] = depInfo.seenOps.insert(op);
      if (!insert) {
        return;
      }

      for (auto iter = depInfo.ops.begin(); true; ++iter) {
        if (iter == depInfo.ops.end()) {
          depInfo.ops.push_back(op);
          break;
        }
        if (iter + 1 == depInfo.ops.end()) {
          depInfo.ops.push_back(op);
          break;
        }
        if (op->isBeforeInBlock(*iter) && !op->isBeforeInBlock(*(iter + 1))) {
          depInfo.ops.insert(iter + 1, op);
          break;
        }
      }

      if (auto regOp = dyn_cast<seq::FirRegOp>(op)) {
        if (scheduledRegs.at(regOp) != partIndex) {
          depInfo.needCrossRegs.insert(regOp);
          return;
        }
      }
      for (auto oprand : op->getOperands()) {
        analyzeValueDepInfoImpl(depInfo, partIndex, oprand);
      }
    } else {
      // Block argument
      depInfo.args.insert(llvm::cast<BlockArgument>(value));
    }
  }
};

/**
 * Stat the weight of the module (number of operation, total bits of operations, bits of register states)
 */
ModuleWeight PartitionPass::statModule(HWModuleOp module, DenseMap<HWModuleOp, ModuleWeight> &cache) {
  if(auto it = cache.find(module); it != cache.end())
    return it->getSecond();

  ModuleWeight ret;
  module.walk([&](Operation *op) {
    if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
      auto childMod =
          dyn_cast<HWModuleOp>(_sc.getDefinition(inst.getModuleNameAttr()));
      auto child = statModule(childMod, cache);
      ret.bitOps += child.bitOps;
      ret.bitRegs += child.bitRegs;
      ret.bitSram += child.bitSram;
      ret.numOps += child.numOps;
    } else if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      auto bitWidth = hw::getBitWidth(reg.getResult().getType());
      if (bitWidth >= 0)
        ret.bitRegs += bitWidth;
    } else {
      for (auto operand : op->getOperands()) {
        auto bitWidth = hw::getBitWidth(operand.getType());
        ret.numOps += 1;
        if (bitWidth >= 0)
          ret.bitOps += bitWidth;
      }
    }
  });

  cache.insert({module, ret});
  return ret;
}

PortInfo PartitionPass::resultToPort(InstanceOp instop, Value innerVal) {
  auto innerMod = dyn_cast<HWModuleOp>(_sc.getDefinition(instop.getModuleNameAttr()));
  auto resultNum = instop.getNumResults();
  for(size_t i = 0; i < resultNum; ++i) {
    auto instRet = instop.getResult(i);
    if(instRet == innerVal) {
      auto output = llvm::find_if(innerMod.getPortList(), [i](PortInfo p) -> bool {
        return p.isOutput() && p.argNum == i;
      });
      assert(output != nullptr);
      return *output;
    }
  }
  // TODO: emit error
  llvm::llvm_unreachable_internal();
}

Value PartitionPass::outputToResult(InstanceOp instop, Value innerVal) {
  auto innerMod = dyn_cast<HWModuleOp>(_sc.getDefinition(instop.getModuleNameAttr()));
  auto outputOps = innerMod.getOps<hw::OutputOp>();
  assert(!outputOps.empty());

  auto outputOp = *outputOps.begin();
  auto outputIdx = llvm::find(outputOp.getOutputs(), innerVal) - outputOp.getOutputs().begin();
  return instop.getResult(outputIdx);
}

SmallVector<Value> PartitionPass::findRelatedInput(InstanceOp instop, Value val) {
  auto innerMod = dyn_cast<HWModuleOp>(_sc.getDefinition(instop.getModuleNameAttr()));
  auto output = resultToPort(instop, val);
  auto relatedInputs = analyzeCombDep(innerMod, output);
  // TODO: change to map
  SmallVector<Value> result;
  result.reserve(relatedInputs.count());

  for(auto relatedInput : relatedInputs.set_bits())
    result.push_back(instop.getInputs()[relatedInput]);
  
  return result;
}

Value PartitionPass::extractFromInst(OpBuilder builder, Value val, InstanceOp instop, IRMapping &extracted) {
  if(auto mapped = extracted.lookupOrNull(val)) return mapped;

  auto mod = dyn_cast<HWModuleOp>(_sc.getDefinition(instop.getModuleNameAttr()));
  auto dep = analyzeCombDep(mod, val);
  if(dep.count() == 0) {
    // Has to be on boundary (if we actually fixed boundary refresh)
    auto newVal = outputToResult(instop, val);
    extracted.map(val, newVal);
    return newVal;
  }

  if(auto defop = val.getDefiningOp()) {
    for(auto operand : defop->getOperands())
      extractFromInst(builder, operand, instop, extracted);
    auto clonedOp = defop->clone(extracted);
    clonedOp = builder.insert(clonedOp);

    for(size_t i = 0; i < defop->getNumResults(); ++i)
      extracted.map(defop->getResult(i), clonedOp->getResult(i));
    // defop->erase();
    return extracted.lookup(val);
  } else {
    // Is input, should not even reach here
    llvm::llvm_unreachable_internal();
  }
}

/**
 * Analyze the combinatory dependencies within a module (input -> any op)
 */
BitVector PartitionPass::analyzeCombDep(HWModuleOp mod, Value val) {
  if(auto defop = val.getDefiningOp()) {
    BitVector result(mod.getNumInputPorts());

    if(dyn_cast<seq::FirRegOp>(defop)) {
      return {};
    } else if(auto it = _comb_deps.find({ mod, val }); it != _comb_deps.end()) {
      return it->getSecond();
    } else if(auto instop = dyn_cast<hw::InstanceOp>(defop)) {
      // Comes from a child module instance. Look for combinatory path within it
      for(auto inputVal : findRelatedInput(instop, val))
        result |= analyzeCombDep(mod, inputVal);

      _comb_deps.insert({{ mod, val }, result});
      return result;
      // TODO: check nullptr
    } else {
      // Ordinary computational operation, cachable
      for(auto operand : defop->getOperands())
        result |= analyzeCombDep(mod, operand);

      _comb_deps.insert({{ mod, val }, result});
      return result;
    }
  } else {
    // Find corresponding port
    for(size_t i = 0; i < mod.getNumInputPorts(); ++i) {
      auto arg = mod.getArgumentForPort(i);
      if(arg == val) {
        BitVector result(mod.getNumInputPorts());
        result.set(i);
        return result;
      }
    }
    // TODO: emit error
    llvm::llvm_unreachable_internal();
  }
}

/**
 * Analyze the combinatory dependencies within a module (input -> output)
 */
BitVector PartitionPass::analyzeCombDep(HWModuleOp module, PortInfo port) {
  assert(port.isOutput());

  auto outputOps = module.getOps<hw::OutputOp>();
  assert(!outputOps.empty());

  auto outputOp = *outputOps.begin();
  auto outputVal = outputOp.getOutputs()[port.argNum];
  return analyzeCombDep(module, outputVal);
}

template<typename T>
struct BFSQueue {
  std::deque<T> bfs;
  DenseSet<T> issued;

  bool push(T t) {
    if(issued.contains(t)) return false;
    issued.insert(t);
    bfs.push_back(t);
    return true;
  }

  std::optional<T> pop() {
    if(bfs.empty()) return {};
    auto ret = bfs.front();
    bfs.pop_front();
    return { ret };
  }
};

/**
 * Transform a module s.t. the specific output ports would no longer depends on
 * input ports combinatorily.
 * 
 * Done in three steps:
 * 1. BFS outside any children modules, using children modules' port dependencies relation
 *   to "jump over" them.
 * 2. Modify module
 * 3. Assemble instance
 */
NormalizationResult PartitionPass::normalizeInst(OpBuilder builder, InstanceOp instop, const BitVector &outputFilter) {
  // TODO: reuse previous results if the extracted ports are the same
  // TODO: normalize output relative to part of input

  auto module = dyn_cast<HWModuleOp>(_sc.getDefinition(instop.getModuleNameAttr()));

  // Step 1, BFS

  // BFS queue
  BFSQueue<Value> bfs;
  // Operations that are to be extracted (has dependency on input)
  DenseSet<Operation *> extracted;
  // Combinatory paths embedded in modules instances
  DenseMap<InstanceOp, BitVector> embeddedPaths;
  // Newly exposed output ports
  DenseSet<Value> boundary;

  auto outputOps = module.getOps<hw::OutputOp>();
  assert(!outputOps.empty());
  auto outputOp = *outputOps.begin();
  SmallVector<Value> origOutputs(outputOp.getOutputs());

  for(size_t i : outputFilter.set_bits()) {
    auto val = origOutputs[i];
    bfs.push(val);
  }

  while(auto valOpt = bfs.pop()) {
    Value val = *valOpt;

    auto dependencies = analyzeCombDep(module, val);
    if(dependencies.count() == 0) {
      // Visisted but does not depend on input, is boundary
      boundary.insert(val);
      continue;
    }

    if(auto defop = val.getDefiningOp()) {
      // Cannot be register
      assert(!dyn_cast<seq::FirRegOp>(defop));

      if(auto instop = dyn_cast<hw::InstanceOp>(defop)) {
        // TODO: corner case: the entire instance has no state, and is extracted outside
        auto output = resultToPort(instop, val);
        embeddedPaths.getOrInsertDefault(instop).set(output.argNum);

        // Find related inputs
        for(auto inputVal : findRelatedInput(instop, val))
          bfs.push(inputVal);
      } else {
        // Ordinary computations
        extracted.insert(defop);
        for(auto operand : defop->getOperands())
          bfs.push(operand);
      }
    } else {
      // Is just input
      continue;
    }
  }

  // Step 2, modify module
  // Step 2.1, normalize all instances
  for(auto &[subinstop, ports] : embeddedPaths) {
    auto normalized = normalizeInst(builder, subinstop, ports);
    subinstop.replaceAllUsesWith(normalized.outputs);
    subinstop.erase();
  }
  // FIXME: refresh boundary

  // Step 2.2, change output to boundary values
  // FIXME: this currently doesn't work if there is multiple instances of a module
  // This is super hacky, and we'd better switch to creating a new module instead.
  // TODO: Is there a way to clone a operation, while mapping a op to a group of op or erase it?
  size_t oldOutputCnt = module.getNumOutputPorts();
  SmallVector<unsigned> erasedOutputs;
  for(size_t i = 0; i < oldOutputCnt; ++i) erasedOutputs.push_back(i);
  module.modifyPorts({}, {}, {}, erasedOutputs);
  outputOp->eraseOperands(0, oldOutputCnt);

  SmallVector<Value> newOutputVals(boundary.begin(), boundary.end());
  size_t ocnt = 0;
  for(const auto &val : newOutputVals) module.appendOutput(std::string("o") + std::to_string(ocnt ++), val);
  builder.setInsertionPoint(outputOp);

  // Step 3, update instance
  builder.setInsertionPoint(instop);
  auto rawInputs = instop.getInputs();
  SmallVector<Value> inputs(rawInputs.begin(), rawInputs.end());
  auto newInstop = builder.create<hw::InstanceOp>(
      instop.getLoc(), module, instop.getInstanceName(), 
      inputs,
      ArrayAttr{}, InnerSymAttr{});
  // Clone anything from excluded into here

  // Replace outputs with newly created ops or result of new inst
  IRMapping mapped;
  for(size_t i = 0; i < instop.getNumOperands(); ++i)
    mapped.map(module.getBodyBlock()->getArgument(i), instop.getOperand(i));
  SmallVector<Value> newOutputs = llvm::map_to_vector(origOutputs, [&](Value origOutput) -> Value {
    return extractFromInst(builder, origOutput, newInstop, mapped);
  });

  return NormalizationResult {
    .instop = newInstop,
    .outputs = newOutputs,
  };
}

DenseMap<HWModuleOp, ModuleWeight> PartitionPass::preparePartition(OpBuilder builder, HWModuleOp mod) {
  // Find all instances
  for(auto instop : mod.getOps<hw::InstanceOp>()) {
    auto innerMod = dyn_cast<HWModuleOp>(_sc.getDefinition(instop.getModuleNameAttr()));
    BitVector outputFilter(innerMod.getNumOutputPorts(), true);
    auto normalized = normalizeInst(builder, instop, outputFilter);
    instop.replaceAllUsesWith(normalized.outputs);
    instop.erase();
  }

  DenseMap<HWModuleOp, ModuleWeight> cache;
  statModule(mod, cache);
  return cache;
}

static void partitionModule(OpBuilder builder, HWModuleOp mod) {
  auto *ctx = builder.getContext();

  auto outputOp = *mod.getOps<hw::OutputOp>().begin();

  ModuleAnalyzer analyzer;
  auto parts = analyzer.scheduleParts(mod);

  for (auto [i, part] : llvm::enumerate(parts)) {
    auto ports = llvm::to_vector(
        llvm::map_range(llvm::enumerate(mod.getPortList()), [](auto port) {
          return std::make_pair(port.index(), port.value());
        }));
    auto outputPorts = SmallVector<std::pair<size_t, hw::PortInfo>>(
        llvm::remove_if(ports,
                        [](auto port) { return port.second.isOutput(); }),
        ports.end());

    auto newPorts =
        llvm::map_to_vector(part.depInfo.args, [&ports](const auto &arg) {
          return ports[arg.getArgNumber()];
        });
    newPorts.push_back(outputPorts[i]);

    // Create a new module for this part.
    builder.setInsertionPoint(mod);
    auto newMod = builder.create<HWModuleOp>(
        mod.getLoc(), builder.getStringAttr(generateModuleName(mod, i)),
        llvm::map_to_vector(newPorts, [](auto port) { return port.second; }));

    // Erase the default created hw.output op.
    (*newMod.getOps<hw::OutputOp>().begin()).erase();

    // Clone body related to this output.
    BackedgeBuilder bb{builder, mod.getLoc()};
    ValueMapper mapper{&bb};
    for (const auto &[i, newPort] : llvm::enumerate(newPorts)) {
      if (newPort.second.isInput()) {
        mapper.set(mod.getBodyBlock()->getArgument(newPort.first),
                   newMod.getBodyBlock()->getArgument(i));
      }
    }
    builder.setInsertionPointToStart(newMod.getBodyBlock());
    for (auto op : llvm::reverse(part.depInfo.ops)) {
      IRMapping bvMapper;
      for (auto operand : op->getOperands()) {
        bvMapper.map(operand, mapper.get(operand));
      }

      auto *newOp = builder.clone(*op, bvMapper);
      for (auto &&[oldRes, newRes] :
           llvm::zip(op->getResults(), newOp->getResults())) {
        mapper.set(oldRes, newRes);
      }
    }

    for (auto reg : part.depInfo.needCrossRegs) {
      auto regInput =
          newMod
              .appendInput(builder.getStringAttr(generateRegPortName(reg)),
                           reg.getResult().getType())
              .second;
      auto newReg = mapper.get(reg);
      newReg.replaceAllUsesWith(regInput);
      newReg.getDefiningOp()->erase();
    }

    builder.create<hw::OutputOp>(
        outputOp.getLoc(), ValueRange{mapper.get(part.depInfo.outputValue)});

    for (auto op : llvm::reverse(part.depInfo.ops)) {
      if (auto regOp = dyn_cast<seq::FirRegOp>(op)) {
        if (analyzer.getPartIndexForReg(regOp) == i) {
          part.ownedRegs[regOp] = newMod.getNumOutputPorts();
          newMod.appendOutput(builder.getStringAttr(generateRegPortName(regOp)),
                              mapper.get(regOp));
        }
      }
    }

    part.newMod = newMod;
  }

  mod.erase();
  return;

  // TODO: Main module vvvvvv

  DenseSet<Operation *> instances;
  for (auto [i, part] : llvm::enumerate(parts)) {
    // Instanciate the new module in the original module.
    auto lastOp = *part.depInfo.ops.begin();
    builder.setInsertionPointAfter(lastOp);

    auto args = part.depInfo.args;
    part.depInfo.needCrossRegs;

    auto inst = builder.create<hw::InstanceOp>(
        mod.getLoc(), part.newMod,
        builder.getStringAttr(std::string{"inst_"} + std::to_string(i)),
        ArrayRef<Value>(part.depInfo.args.getArrayRef().begin(),
                        part.depInfo.args.getArrayRef().end()),
        ArrayAttr{}, InnerSymAttr{});
    lastOp->getResult(0).replaceAllUsesWith(inst.getResult(0));

    instances.insert(inst);
    part.newInst = inst;
  }

  mod.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      [&](Operation *op) {
        if (auto regOp = dyn_cast<seq::FirRegOp>(op)) {
          auto partIndex = analyzer.getPartIndexForReg(regOp);
          auto regOut = parts[partIndex].newInst.getResult(0);
          regOp.replaceAllUsesWith(regOut);
          regOp.erase();
        }
      });
  mod.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      [&](Operation *op) {
        if (!instances.contains(op) && !isa<hw::OutputOp>(*op) &&
            op != mod.getOperation()) {
          op->erase();
        }
      });
}

PartitionPlan PartitionPass::partition(OpBuilder builder, HWModuleOp module) {
  // FIXME: Currently let's just randomly partition them.
  PartitionPlan plan;
  std::mt19937 gen(0xdeadbeef);
  std::uniform_int_distribution<size_t> dist(0, parts - 1);
  module.walk([&](Operation *op) {
    plan.fwdMap.insert({ op, dist(gen) });
  });
  return plan;
}

struct PendingRegInput {
  Backedge placeholder;
  size_t regChunk;
};

void PartitionPass::commitPartition(OpBuilder builder, HWModuleOp root, PartitionPlan plan) {
  size_t crossPortCnt = 0;
  BackedgeBuilder bb{builder, root.getLoc()};

  DenseMap<Value, PortInfo> crossOut;
  SmallVector<IRMapping> localMapping;
  DenseMap<Operation *, Operation *> cloned;
  DenseMap<std::pair<Value, size_t>, Backedge> pendingRegs;

  localMapping.resize(parts);

  // Create all modules
  SmallVector<HWModuleOp> results;
  builder.setInsertionPointAfter(root);
  for(size_t i = 0; i < parts; ++i) {
    SmallVector<PortInfo> initPorts;
    for(auto port : root.getPortList()) if(port.isInput()) initPorts.push_back(port);

    auto mod = builder.create<HWModuleOp>(
      root.getLoc(), builder.getStringAttr(root.getName() + "_" + std::to_string(i)), initPorts
    );

    auto &mapping = localMapping[i];
    for(size_t in = 0; in < initPorts.size(); ++in)
      mapping.map(root.getArgumentForInput(in), mod.getArgumentForInput(in));

    results.push_back(mod);
  }

  // Map all inputs

  const auto dfsClone = [&](Operation *op) {
    auto impl = [&](Operation *op, auto &dfs) mutable {
      if(isa<OutputOp>(op)) return;
      if(cloned.contains(op)) return;

      auto chunk = plan.fwdMap.find(op)->second;
      auto localMod = results[chunk];
      bool isReg = isa<seq::FirRegOp>(op);
      SmallVector<std::optional<Backedge>> regInputBackedges;
      if(isReg) regInputBackedges.resize(op->getNumOperands());
      bool hasBackedge = false;
      auto &mapping = localMapping[chunk];

      for(size_t operandIdx = 0; operandIdx < op->getNumOperands(); ++operandIdx) {
        auto operand = op->getOperand(operandIdx);
        if(mapping.contains(operand)) // Already reachable here
          continue;

        // Currently not reachable
        if(auto defop = operand.getDefiningOp()) {
          if(isReg && !cloned.contains(defop)) {
            // Reg dependency not created yet, generate a backedge
            // In this case we skip DFS or we risk a infinite loop
            Backedge backedge;
            auto backedgeLookup = pendingRegs.find({ operand, chunk });
            if(backedgeLookup != pendingRegs.end()) {
              backedge = backedgeLookup->second;
            } else {
              backedge = bb.get(operand.getType());
              pendingRegs.insert({
                { operand, chunk }, backedge
              });
            }
            
            regInputBackedges[operandIdx] = backedge;
            hasBackedge = true;
            continue;
          }

          dfs(defop, dfs);
          auto clonedDefop = cloned.find(defop)->second;
          auto remoteChunk = plan.fwdMap.find(defop)->second;

          auto resultIdx = llvm::find(defop->getResults(), operand) - defop->getResults().begin();
          auto newOperand = clonedDefop->getResult(resultIdx);

          if(remoteChunk == chunk) {
            // Trivially reachable
            mapping.map(operand, newOperand);
            defop->emitWarning()<<" operand "<<operandIdx<<" is locally reachable: "<<newOperand;
          } else {
            // In remote module, create cross ports if didn't exist
            PortInfo port;
            if(auto lookup = crossOut.find(newOperand); lookup != crossOut.end())
              port = lookup->second;
            else {
              auto ticket = crossPortCnt++;
              auto remoteMod = results[remoteChunk];
              remoteMod.appendOutput(
                builder.getStringAttr(std::string("cross_") + std::to_string(ticket)),
                newOperand
              );
              port = remoteMod.getPort(remoteMod.getNumPorts() - 1);
              crossOut.insert({ newOperand, port });
            }

            auto [_, localVal] = localMod.appendInput(port.getName(), port.type);
            mapping.map(operand, localVal);
          }
        } else {
          // All inputs are pre-mapped, this cannot happend
          llvm::llvm_unreachable_internal();
        }
      }

      // Operands are fully resolved.
      builder.setInsertionPointToStart(localMod.getBodyBlock());
      Operation *clonedOp;
      if(hasBackedge) {
        IRMapping tmpMapping = mapping;
        for(size_t operandIdx = 0; operandIdx < op->getNumOperands(); ++operandIdx)
          if(auto backedge = regInputBackedges[operandIdx]) {
            tmpMapping.map(op->getOperand(operandIdx), *backedge);
          }
        clonedOp = builder.clone(*op, tmpMapping);
      } else {
        clonedOp = builder.clone(*op, mapping);
      }

      cloned.insert({ op, clonedOp });
    };
    impl(op, impl);
  };


  root.getBodyBlock()->walk(dfsClone);

  // Resolve all unresolved reg backedges
  for(auto [key, backedge] : pendingRegs) {
    auto [val, chunk] = key;
    auto defop = val.getDefiningOp();
    auto clonedDefop = cloned.find(defop)->second;
    auto remoteChunk = plan.fwdMap.find(defop)->second;

    auto resultIdx = llvm::find(defop->getResults(), val) - defop->getResults().begin();
    auto newOperand = clonedDefop->getResult(resultIdx);

    if(remoteChunk == chunk) {
      // Trivially connectable
      backedge.setValue(newOperand);
    } else {
      auto localMod = results[chunk];

      // TODO: dedup this chunk ot code. This is exactly the same as above
      PortInfo port;
      if(auto lookup = crossOut.find(newOperand); lookup != crossOut.end())
        port = lookup->second;
      else {
        auto ticket = crossPortCnt++;
        auto remoteMod = results[remoteChunk];
        remoteMod.appendOutput(
          builder.getStringAttr(std::string("cross_") + std::to_string(ticket)),
          newOperand
        );
        port = remoteMod.getPort(remoteMod.getNumPorts() - 1);
        crossOut.insert({ newOperand, port });
      }

      auto [_, localVal] = localMod.appendInput(port.getName(), port.type);
      backedge.setValue(localVal);
    }
  }
}

void PartitionPass::runOnOperation() {
  auto builder = OpBuilder{&getContext()};

  auto root = getOperation();
  _sc.addDefinitions(root);

  auto modules = getOperation().getOps<HWModuleOp>();
  HWModuleOp selected;

  if(moduleName == "") {
    auto filtered = llvm::find_if(modules, [](HWModuleOp mod) -> bool { return mod.isPublic(); });
    bool failed = false;
    if(filtered == modules.end()) failed = true;
    else {
      selected = *filtered;
      ++filtered;
      if(filtered != modules.end()) failed = true;
    }
    if(failed) {
      emitError(getOperation().getLoc(), "Less or more than one public module found. Please specify module name"); // Here
      signalPassFailure();
      return;
    }
  } else {
    auto filtered = llvm::find_if(modules, [&](auto hwModule) { return hwModule.getName() == moduleName; });
    if (filtered == modules.end()) {
      emitError(getOperation().getLoc(), "module '" + moduleName + "' not found"); // Here
      signalPassFailure();
      return;
    }
    selected = *filtered;
  }


  preparePartition(builder, selected);
  auto plan = partition(builder, selected);
  commitPartition(builder, selected, plan);
  // partitionModule(builder, *module);
}

std::unique_ptr<Pass> circt::hw::createPartitionPass(std::string moduleName) {
  return std::make_unique<PartitionPass>(std::move(moduleName));
}