//===- InferMemories.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Transform CHIRRTL memory operations and memory ports into standard FIRRTL
// memory operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

namespace {
struct InferMemoriesPass : public InferMemoriesBase<InferMemoriesPass>,
                           public FIRRTLVisitor<InferMemoriesPass> {

  using FIRRTLVisitor<InferMemoriesPass>::visitDecl;
  using FIRRTLVisitor<InferMemoriesPass>::visitExpr;
  using FIRRTLVisitor<InferMemoriesPass>::visitStmt;

  void visitDecl(CombMemOp op);
  void visitDecl(SeqMemOp op);
  void visitStmt(MemoryPortOp op);
  void visitStmt(MemoryPortAccessOp op);
  void visitExpr(SubaccessOp op);
  void visitExpr(SubfieldOp op);
  void visitExpr(SubindexOp op);
  void visitStmt(ConnectOp op);
  void visitStmt(PartialConnectOp op);
  void visitUnhandledOp(Operation *op);

  /// Get a the constant 0.  This constant is inserted at the beginning of the
  /// module.
  Value getConst(unsigned c) {
    auto &value = constCache[c];
    if (!value) {
      auto module = getOperation();
      auto builder = OpBuilder::atBlockBegin(module.getBodyBlock());
      auto u1Type = UIntType::get(builder.getContext(), /*width*/ 1);
      value = builder.create<ConstantOp>(module.getLoc(), u1Type,
                                         APInt(/*bitWidth*/ 1, c, false));
    }
    return value;
  }

  //// Clear out any stale data.
  void clear() {
    constCache.clear();
    invalidCache.clear();
    opsToDelete.clear();
    subfieldDirs.clear();
    rdataValues.clear();
    wdataValues.clear();
  }

  void emitInvalid(ImplicitLocOpBuilder &builder, Value value);

  MemDirAttr inferMemoryPortKind(MemoryPortOp memPort);

  void replaceMem(Operation *op, StringRef name, bool isSequential, RUWAttr ruw,
                  ArrayAttr annotations);

  template <typename OpType, typename... T>
  void cloneSubindexOpForMemory(OpType op, Value input, T... operands);

  void emitPartialConnectMask(
      ImplicitLocOpBuilder &builder, Type destType, Type srcType,
      llvm::function_ref<Value(ImplicitLocOpBuilder &)> getSubAccess);

  void runOnOperation() override;

  /// Cached constants.
  DenseMap<unsigned, Value> constCache;
  DenseMap<Type, Value> invalidCache;

  /// List of operations to delete at the end of the pass.
  SmallVector<Operation *> opsToDelete;

  /// This tracks how the result of a subfield operation which is indexes a
  /// MemoryPortOp is used.  This is used to track if the subfield operation
  /// needs to be cloned to access a memories rdata or wdata.
  DenseMap<Operation *, MemDirAttr> subfieldDirs;

  /// This maps a subfield-like operation from a MemoryPortOp to a new subfield
  /// operation which can be used to read from the MemoryOp. This is used to
  /// update any operations to read from the new memory.
  DenseMap<Value, Value> rdataValues;

  /// This maps a subfield-like operation from a MemoryPortOp to a new subfield
  /// operation which can be used to write to the memory, the mask value which
  /// should be set to 1, and the the correspoding wmode port of the memory
  /// which should be set to 1.  Not all memories have wmodes, so this field
  /// can be null. This is used to update operations to write to the new memory.
  struct WDataInfo {
    Value data;
    Value mask;
    Value mode;
  };
  DenseMap<Value, WDataInfo> wdataValues;
};
} // end anonymous namespace

/// Performs the callback for each leaf element of a value.  This will create
/// any subindex and subfield operations needed to access the leaf values of the
/// aggregate value.
static void forEachLeaf(ImplicitLocOpBuilder &builder, Value value,
                        llvm::function_ref<void(Value)> func) {
  auto type = value.getType();
  if (auto bundleType = type.dyn_cast<BundleType>()) {
    for (size_t i = 0, e = bundleType.getNumElements(); i < e; ++i)
      forEachLeaf(builder, builder.create<SubfieldOp>(value, i), func);
  } else if (auto vectorType = type.dyn_cast<FVectorType>()) {
    for (size_t i = 0, e = vectorType.getNumElements(); i != e; ++i)
      forEachLeaf(builder, builder.create<SubindexOp>(value, i), func);
  } else {
    func(value);
  }
}

/// Drive a value to all leafs of the input aggregate value. This only makes
/// sense when all leaf values have the same type, since the same value will be
/// connected to each leaf. This does not work for aggregates with flip types.
static void connectLeafsTo(ImplicitLocOpBuilder &builder, Value bundle,
                           Value value) {
  forEachLeaf(builder, bundle,
              [&](Value leaf) { builder.create<ConnectOp>(leaf, value); });
}

/// Connect each leaf of an aggregate type to invalid.  This does not support
/// aggregates with flip types.
void InferMemoriesPass::emitInvalid(ImplicitLocOpBuilder &builder,
                                    Value value) {
  auto type = value.getType();
  auto &invalid = invalidCache[type];
  if (!invalid) {
    auto builder = OpBuilder::atBlockBegin(getOperation().getBodyBlock());
    invalid = builder.create<InvalidValueOp>(getOperation().getLoc(), type);
  }
  builder.create<ConnectOp>(value, invalid);
}

/// Converts a CHIRRTL memory port direction to a MemoryOp port type.  The
/// biggest difference is that there is no match for the Infer port type.
static MemOp::PortKind memDirAttrToPortKind(MemDirAttr direction) {
  switch (direction) {
  case MemDirAttr::Read:
    return MemOp::PortKind::Read;
  case MemDirAttr::Write:
    return MemOp::PortKind::Write;
  case MemDirAttr::ReadWrite:
    return MemOp::PortKind::ReadWrite;
  default:
    llvm_unreachable(
        "Unhandled MemDirAttr, was the port direction not inferred?");
  }
}

/// This function infers the memory direction of each CHIRRTL memory port. Each
/// memory port has an initial memory direction which is explicitly declared in
/// the MemoryPortOp, which is used as a starting point.  For example, if the
/// port is declared to be Write, but it is only ever read from, the port will
/// become a ReadWrite port.
///
/// When the memory port is eventually replaced with a memory, we will go from
/// having a single data value to having separate rdata and wdata values.  In
/// this function we record how the result of each data subfield operation is
/// used, so that later on we can make sure the SubfieldOp is cloned to index
/// into the correct rdata and wdata fields of the memory.
MemDirAttr InferMemoriesPass::inferMemoryPortKind(MemoryPortOp memPort) {
  // This function does a depth-first walk of the use-lists of the memport
  // operation to look through subindex operations and find the places where it
  // is ultimately used.  At each node we record how the children ops are using
  // the result of the current operation.  When we are done visiting the current
  // operation we store how it is used into a global hashtable for later use.
  // This records how both the MemoryPort and Subfield operations are used.
  struct StackElement {
    StackElement(Value value, Value::use_iterator iterator, MemDirAttr mode)
        : value(value), iterator(iterator), mode(mode) {}
    Value value;
    Value::use_iterator iterator;
    MemDirAttr mode;
  };

  SmallVector<StackElement> stack;
  stack.emplace_back(memPort.data(), memPort.data().use_begin(),
                     memPort.direction());
  MemDirAttr mode = MemDirAttr::Infer;

  while (!stack.empty()) {
    auto *iter = &stack.back().iterator;
    auto end = stack.back().value.use_end();
    stack.back().mode |= mode;

    while (*iter != end) {
      auto &element = stack.back();
      auto &use = *(*iter);
      auto *user = use.getOwner();
      ++(*iter);
      if (isa<SubindexOp, SubfieldOp>(user)) {
        // We recurse into Subindex ops to find the leaf-uses.
        auto input = user->getResult(0);
        stack.emplace_back(input, input.use_begin(), MemDirAttr::Infer);
        mode = MemDirAttr::Infer;
        iter = &stack.back().iterator;
        end = input.use_end();
        continue;
      }
      if (auto subaccessOp = dyn_cast<SubaccessOp>(user)) {
        // Subaccess has two arguments, the vector and the index. If we are
        // using the memory port as an index, we can ignore it. If we are using
        // the memory as the vector, we need to recurse.
        auto input = subaccessOp.input();
        if (use.get() == input) {
          stack.emplace_back(input, input.use_begin(), MemDirAttr::Infer);
          mode = MemDirAttr::Infer;
          iter = &stack.back().iterator;
          end = input.use_end();
          continue;
        }
        // Otherwise we are reading from a memory for the index.
        element.mode |= MemDirAttr::Read;
      } else if (auto connectOp = dyn_cast<ConnectOp>(user)) {
        if (use.get() == connectOp.dest()) {
          element.mode |= MemDirAttr::Write;
        } else {
          element.mode |= MemDirAttr::Read;
        }
      } else if (auto partialConnectOp = dyn_cast<PartialConnectOp>(user)) {
        if (use.get() == partialConnectOp.dest()) {
          element.mode |= MemDirAttr::Write;
        } else {
          element.mode |= MemDirAttr::Read;
        }
      } else {
        // Every other use of a memory is a read operation.
        element.mode |= MemDirAttr::Read;
      }
    }
    mode = stack.back().mode;

    // Store the direction of the current operation in the global map. This will
    // be used later to determine if this subaccess operation needs to be cloned
    // into rdata, wdata, and wmask.
    subfieldDirs[stack.back().value.getDefiningOp()] = mode;
    stack.pop_back();
  }

  return mode;
}

void InferMemoriesPass::replaceMem(Operation *cmem, StringRef name,
                                   bool isSequential, RUWAttr ruw,
                                   ArrayAttr annotations) {
  assert(isa<CombMemOp>(cmem) || isa<SeqMemOp>(cmem));

  // We have several early breaks in this function, so we record the CHIRRTL
  // memory for deletion here.
  opsToDelete.push_back(cmem);

  auto cmemType = cmem->getResult(0).getType().cast<CMemoryType>();
  auto depth = cmemType.getNumElements();
  auto type = cmemType.getElementType();

  // Collect the information from each of the CMemoryPorts.
  struct PortInfo {
    StringAttr name;
    Type type;
    Attribute annotations;
    MemOp::PortKind portKind;
    MemoryPortOp cmemPort;
  };
  SmallVector<PortInfo, 4> ports;
  for (auto *user : cmem->getUsers()) {
    auto cmemoryPort = cast<MemoryPortOp>(user);

    // Infer the type of memory port we need to create.
    auto portDirection = inferMemoryPortKind(cmemoryPort);

    // If the memory port is never used, it will have the Infer type and should
    // just be deleted. TODO: this is mirroring SFC, but should we be checking
    // for annotations on the memory port before removing it?
    if (portDirection == MemDirAttr::Infer)
      continue;
    auto portKind = memDirAttrToPortKind(portDirection);

    // Add the new port.
    ports.push_back({cmemoryPort.nameAttr(),
                     MemOp::getTypeForPort(depth, type, portKind),
                     cmemoryPort.annotationsAttr(), portKind, cmemoryPort});
  }

  // If there are no valid memory ports, don't create a memory.
  if (ports.empty())
    return;

  // Canonicalize the ports into alphabetical order.
  llvm::array_pod_sort(ports.begin(), ports.end(),
                       [](const PortInfo *lhs, const PortInfo *rhs) -> int {
                         return lhs->name.getValue().compare(
                             rhs->name.getValue());
                       });

  SmallVector<Attribute, 4> resultNames;
  SmallVector<Type, 4> resultTypes;
  SmallVector<Attribute, 4> portAnnotations;
  for (auto port : ports) {
    resultNames.push_back(port.name);
    resultTypes.push_back(port.type);
    portAnnotations.push_back(port.annotations);
  }

  // Write latency is always 1, while the read latency depends on the memory
  // type.
  auto readLatency = isSequential ? 1 : 0;
  auto writeLatency = 1;

  // Create the memory.
  ImplicitLocOpBuilder memBuilder(cmem->getLoc(), cmem);
  auto memory = memBuilder.create<MemOp>(
      resultTypes, readLatency, writeLatency, depth, ruw,
      memBuilder.getArrayAttr(resultNames), name, annotations,
      memBuilder.getArrayAttr(portAnnotations));

  // Process each memory port, initializing the memory port and inferring when
  // to set the enable signal high.
  for (unsigned i = 0, e = memory.getNumResults(); i < e; ++i) {
    auto cmemoryPort = ports[i].cmemPort;
    auto cmemoryPortAccess = cmemoryPort.getAccess();
    auto memoryPort = memory.getResult(i);
    auto portKind = ports[i].portKind;

    // Most fields on the newly created memory will be assigned an initial value
    // immediately following the memory decl, and then will be assigned a second
    // value at the location of the CHIRRTL memory port.

    // Initialization at the MemoryOp.
    ImplicitLocOpBuilder portBuilder(cmemoryPortAccess.getLoc(),
                                     cmemoryPortAccess);
    auto address = memBuilder.create<SubfieldOp>(memoryPort, "addr");
    emitInvalid(memBuilder, address);
    auto enable = memBuilder.create<SubfieldOp>(memoryPort, "en");
    memBuilder.create<ConnectOp>(enable, getConst(0));
    auto clock = memBuilder.create<SubfieldOp>(memoryPort, "clk");
    emitInvalid(memBuilder, clock);

    // Initialization at the MemoryPortOp
    portBuilder.create<ConnectOp>(address, cmemoryPortAccess.index());
    // Sequential+Read ports have a more complicated "enable inference".
    // Everything else sets the enable to true.
    if (!(portKind == MemOp::PortKind::Read && isSequential)) {
      portBuilder.create<ConnectOp>(enable, getConst(1));
    }
    portBuilder.create<ConnectOp>(clock, cmemoryPortAccess.clock());

    if (portKind == MemOp::PortKind::Read) {
      // Store the read information for updating subfield ops.
      auto data = memBuilder.create<SubfieldOp>(memoryPort, "data");
      rdataValues[cmemoryPort.data()] = data;
    } else if (portKind == MemOp::PortKind::Write) {
      // Initialization at the MemoryOp.
      auto data = memBuilder.create<SubfieldOp>(memoryPort, "data");
      emitInvalid(memBuilder, data);
      auto mask = memBuilder.create<SubfieldOp>(memoryPort, "mask");
      emitInvalid(memBuilder, mask);

      // Initialization at the MemoryPortOp.
      connectLeafsTo(portBuilder, mask, getConst(0));

      // Store the write information for updating subfield ops.
      wdataValues[cmemoryPort.data()] = {data, mask, nullptr};
    } else if (portKind == MemOp::PortKind::ReadWrite) {
      // Initialization at the MemoryOp.
      auto rdata = memBuilder.create<SubfieldOp>(memoryPort, "rdata");
      auto wmode = memBuilder.create<SubfieldOp>(memoryPort, "wmode");
      memBuilder.create<ConnectOp>(wmode, getConst(0));
      auto wdata = memBuilder.create<SubfieldOp>(memoryPort, "wdata");
      emitInvalid(memBuilder, wdata);
      auto wmask = memBuilder.create<SubfieldOp>(memoryPort, "wmask");
      emitInvalid(memBuilder, wmask);

      // Initialization at the MemoryPortOp.
      connectLeafsTo(portBuilder, wmask, getConst(0));

      // Store the read and write information for updating subfield ops.
      wdataValues[cmemoryPort.data()] = {wdata, wmask, wmode};
      rdataValues[cmemoryPort.data()] = rdata;
    }

    // Sequential read only memory ports have "enable inference", which
    // detects when to set the enable high. All other memory ports set the
    // enable high when the memport is declared. This is higly questionable
    // logic that is easily defeated. This behaviour depends on the kind of
    // operation used as the memport index.
    if (portKind == MemOp::PortKind::Read && isSequential) {
      auto *indexOp = cmemoryPortAccess.index().getDefiningOp();
      bool success = false;
      if (!indexOp) {
        // TODO: SFC does not infer any enable when using a module port as the
        // address.  This seems like something that should be fixed sooner
        // rather than later.
      } else if (isa<WireOp, RegResetOp, RegOp>(indexOp)) {
        // If the address is a reference, then we set the enable whenever the
        // address is driven.

        // Find the uses of the address that write a value to it, ignoring the
        // ones driving an invalid value.
        auto drivers =
            make_filter_range(indexOp->getUsers(), [&](Operation *op) {
              if (auto connectOp = dyn_cast<ConnectOp>(op)) {
                if (cmemoryPortAccess.index() == connectOp.dest())
                  return !dyn_cast_or_null<InvalidValueOp>(
                      connectOp.src().getDefiningOp());
              } else if (auto pConnectOp = dyn_cast<PartialConnectOp>(op)) {
                if (cmemoryPortAccess.index() == pConnectOp.dest())
                  return !dyn_cast_or_null<InvalidValueOp>(
                      pConnectOp.src().getDefiningOp());
              }
              return false;
            });

        // At each location where we drive a value to the index, set the enable.
        for (auto *driver : drivers) {
          OpBuilder(driver).create<ConnectOp>(driver->getLoc(), enable,
                                              getConst(1));
          success = true;
        }
      } else if (isa<NodeOp>(indexOp)) {
        // If using a Node for the address, then the we place the enable at the
        // Node op's
        OpBuilder(indexOp).create<ConnectOp>(indexOp->getLoc(), enable,
                                             getConst(1));
        success = true;
      }

      // If we don't infer any enable points, it is almost always a user error.
      if (!success)
        cmemoryPort.emitWarning("memory port is never enabled");
    }
  }
}

void InferMemoriesPass::visitDecl(CombMemOp combmem) {
  replaceMem(combmem, combmem.name(), /*isSequential*/ false,
             RUWAttr::Undefined, combmem.annotations());
}

void InferMemoriesPass::visitDecl(SeqMemOp seqmem) {
  replaceMem(seqmem, seqmem.name(), /*isSequential*/ true, seqmem.ruw(),
             seqmem.annotations());
}

void InferMemoriesPass::visitStmt(MemoryPortOp memPort) {
  // The memory port is mostly handled while processing the memory.
  opsToDelete.push_back(memPort);
}

void InferMemoriesPass::visitStmt(MemoryPortAccessOp memPortAccess) {
  // The memory port access is mostly handled while processing the memory.
  opsToDelete.push_back(memPortAccess);
}

void InferMemoriesPass::visitStmt(ConnectOp connect) {
  // Check if we are writing to a memory and, if we are, replace the
  // destination.
  auto writeIt = wdataValues.find(connect.dest());
  if (writeIt != wdataValues.end()) {
    auto writeData = writeIt->second;
    connect.destMutable().assign(writeData.data);
    // Assign the write mask.
    ImplicitLocOpBuilder builder(connect.getLoc(), connect);
    connectLeafsTo(builder, writeData.mask, getConst(1));
    // Only ReadWrite memories have a write mode.
    if (writeData.mode)
      builder.create<ConnectOp>(writeData.mode, getConst(1));
  }
  // Check if we are reading from a memory and, if we are, replace the
  // source.
  auto readIt = rdataValues.find(connect.src());
  if (readIt != rdataValues.end()) {
    auto newSource = readIt->second;
    connect.srcMutable().assign(newSource);
  }
}

/// This will find which fields of an aggregate type are connected by a partial
/// connect and connect the same field of the mask to 1. This function is
/// recursive over the types of the destination and source of the partial
/// connect. This uses lambdas to lazily emit subfield operations on the mask
/// only when there is a valid pair-wise connection point.
void InferMemoriesPass::emitPartialConnectMask(
    ImplicitLocOpBuilder &builder, Type destType, Type srcType,
    llvm::function_ref<Value(ImplicitLocOpBuilder &)> getSubaccess) {
  if (auto destBundle = destType.dyn_cast<BundleType>()) {
    // Partial connect will connect together any two fields with the same name.
    // The verifier will have checked that the fields have the same type.
    auto srcBundle = srcType.cast<BundleType>();
    auto end = destBundle.getNumElements();
    for (unsigned destIndex = 0; destIndex < end; ++destIndex) {
      // Try to find a field with the same name in the srcType.
      auto fieldName = destBundle.getElements()[destIndex].name.getValue();
      auto srcIndex = srcBundle.getElementIndex(fieldName);
      if (!srcIndex)
        continue;
      auto &destElt = destBundle.getElements()[destIndex];
      auto &srcElt = srcBundle.getElements()[*srcIndex];

      // Call back to lazily create the subfield op on the mask.
      Value subfield = nullptr;
      auto lazySubfield = [&](ImplicitLocOpBuilder &builder) {
        if (!subfield)
          subfield =
              builder.create<SubfieldOp>(getSubaccess(builder), destIndex);
        return subfield;
      };

      // Recursively handle the connection point.
      emitPartialConnectMask(builder, destElt.type, srcElt.type, lazySubfield);
    }
  } else if (auto destVector = destType.dyn_cast<FVectorType>()) {
    // Partial connect will connect all elements of the vectors together up to
    // the length of the shorter vector.  This needs to recurse for each pair of
    // connected elements.
    auto srcVector = srcType.dyn_cast<FVectorType>();
    auto destEltType = destVector.getElementType();
    auto srcEltType = srcVector.getElementType();
    auto end =
        std::min(destVector.getNumElements(), srcVector.getNumElements());

    for (unsigned i = 0; i < end; i++) {
      // Call back to lazily create the subindex op on the mask.
      Value subindex = nullptr;
      auto lazySubindex = [&](ImplicitLocOpBuilder builder) {
        if (!subindex)
          subindex = builder.create<SubindexOp>(getSubaccess(builder), i);
        return subindex;
      };

      // Recursively handle the connection point.
      emitPartialConnectMask(builder, destEltType, srcEltType, lazySubindex);
    }
  } else {
    // Connect the mask to 1, forcing the creation of any required subfield and
    // subindex operations.
    builder.create<ConnectOp>(getSubaccess(builder), getConst(1));
  }
}

void InferMemoriesPass::visitStmt(PartialConnectOp partialConnect) {
  // Check if we are writing to a memory and, if we are, replace the
  // destination.
  auto writeIt = wdataValues.find(partialConnect.dest());
  if (writeIt != wdataValues.end()) {
    auto writeData = writeIt->second;

    // Update the destination to use the new memory.
    partialConnect.destMutable().assign(writeData.data);

    // Handle the partial connect write mask.  This only sets the mask
    // for the elements which are connected by the partial connect.
    ImplicitLocOpBuilder builder(partialConnect.getLoc(), partialConnect);
    emitPartialConnectMask(
        builder, partialConnect.dest().getType(),
        partialConnect.src().getType(),
        [&](ImplicitLocOpBuilder &builder) { return writeData.mask; });

    // Only ReadWrite memories have a write mode, so this field can sometimes be
    // null.
    if (writeData.mode)
      builder.create<ConnectOp>(writeData.mode, getConst(1));
  }

  // Check if we are reading from a memory and, if we are, replace the
  // source.
  auto readIt = rdataValues.find(partialConnect.src());
  if (readIt != rdataValues.end()) {
    auto newSource = readIt->second;
    // Update the source to use the new memory.
    partialConnect.srcMutable().assign(newSource);
  }
}

/// This function will create clones of subaccess, subindex, and subfield
/// operations which are indexing a CHIRRTL memory ports that will index into
/// the new memory's data field.  If a subfield result is used to read from a
/// memory port, it will be cloned to read from the memory's rdata field.  If
/// the subfield is used to write to a memory port, it will be cloned twice to
/// write to both the wdata and wmask fields. Users of this subfield operation
/// will be redirected to the appropriate clone when they are visited.
template <typename OpType, typename... T>
void InferMemoriesPass::cloneSubindexOpForMemory(OpType op, Value input,
                                                 T... operands) {
  // If the subaccess operation has no direction recorded, then it does not
  // index a CHIRRTL memory and will be left alone.
  auto it = subfieldDirs.find(op);
  if (it == subfieldDirs.end())
    return;

  // All uses of this op will be updated to use the appropriate clone.  If the
  // recorded direction of this subfield is Infer, then the value is not
  // actually used to read or write from a memory port, and it will be just
  // removed.
  opsToDelete.push_back(op);

  auto direction = it->second;
  ImplicitLocOpBuilder builder(op->getLoc(), op);

  // If the subaccess operation is used to read from a memory port, we need to
  // clone it to read from the rdata field.
  if (direction == MemDirAttr::Read || direction == MemDirAttr::ReadWrite) {
    rdataValues[op] = builder.create<OpType>(rdataValues[input], operands...);
  }

  // If the subaccess operation is used to write to the memory, we need to clone
  // it to write to the the wdata and the wmask fields.
  if (direction == MemDirAttr::Write || direction == MemDirAttr::ReadWrite) {
    auto writeData = wdataValues[input];
    auto write = builder.create<OpType>(writeData.data, operands...);
    auto mask = builder.create<OpType>(writeData.mask, operands...);
    wdataValues[op] = {write, mask, writeData.mode};
  }
}

void InferMemoriesPass::visitExpr(SubaccessOp subaccess) {
  // Check if the subaccess reads from a memory for
  // the index.
  auto readIt = rdataValues.find(subaccess.index());
  if (readIt != rdataValues.end()) {
    subaccess.indexMutable().assign(readIt->second);
  }
  // Handle it like normal.
  cloneSubindexOpForMemory(subaccess, subaccess.input(), subaccess.index());
}

void InferMemoriesPass::visitExpr(SubfieldOp subfield) {
  cloneSubindexOpForMemory<SubfieldOp>(subfield, subfield.input(),
                                       subfield.fieldIndex());
}

void InferMemoriesPass::visitExpr(SubindexOp subindex) {
  cloneSubindexOpForMemory<SubindexOp>(subindex, subindex.input(),
                                       subindex.index());
}

void InferMemoriesPass::visitUnhandledOp(Operation *op) {
  // For every operand, check if it is reading from a memory port and
  // replace it with a read from the new memory.
  for (auto &operand : op->getOpOperands()) {
    auto it = rdataValues.find(operand.get());
    if (it != rdataValues.end()) {
      operand.set(it->second);
    }
  }
}

void InferMemoriesPass::runOnOperation() {
  // Walk the entire body of the module and dispatch the visitor on each
  // function.  This will replace all CHIRRTL memories and ports, and update all
  // uses.
  getOperation().getBodyBlock()->walk(
      [&](Operation *op) { dispatchVisitor(op); });

  // If there are no operations to delete, then we didn't find any CHIRRTL
  // memories.
  if (opsToDelete.empty())
    markAllAnalysesPreserved();

  // Remove the old memories and their ports.
  while (!opsToDelete.empty())
    opsToDelete.pop_back_val()->erase();

  // Clear out any cached data.
  clear();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInferMemoriesPass() {
  return std::make_unique<InferMemoriesPass>();
}
