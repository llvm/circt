//===- InferReadWrite.cpp - Infer Read Write Memory -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InferReadWrite pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-infer-read-write"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INFERREADWRITE
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
struct InferReadWritePass
    : public circt::firrtl::impl::InferReadWriteBase<InferReadWritePass> {

  /// This pass performs two memory transformations:
  ///  1. If the multi-bit enable port is connected to a constant 1,
  ///     then, replace with a single bit mask. Create a new memory with a
  ///     1 bit mask, and replace the old memory with it. The single bit mask
  ///     memory is always lowered to an unmasked memory.
  ///  2. If the read and write enable ports are trivially mutually exclusive,
  ///     then create a new memory with a single read/write port, and replace
  ///     the old memory with it.
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "\n Running Infer Read Write on module:"
                            << getOperation().getName());
    SmallVector<Operation *> opsToErase;

    auto result = getOperation().walk([&](Operation *op) {
      // This pass is going to have problems if it tries to determine signal
      // drivers in the presence of WhenOps.  Conservatively error out if we see
      // any WhenOps.
      //
      // TODO: This would be better handled if WhenOps were moved into the
      // CHIRRTL dialect so that this pass could more strongly specify that it
      // only works on FIRRTL as opposed to a subset of FIRRTL.
      if (isa<WhenOp>(op)) {
        op->emitOpError()
            << "is unsupported by InferReadWrite as this pass cannot trace "
               "signal drivers in their presence. Please run `ExpandWhens` to "
               "remove these operations before running this pass.";
        return WalkResult::interrupt();
      }

      MemOp memOp = dyn_cast<MemOp>(op);
      if (!memOp)
        return WalkResult::advance();

      inferUnmasked(memOp, opsToErase);
      simplifyWmode(memOp);
      size_t nReads, nWrites, nRWs, nDbgs;
      memOp.getNumPorts(nReads, nWrites, nRWs, nDbgs);
      // Run the analysis only for Seq memories (latency=1) and a single read
      // and write ports.
      if (!(nReads == 1 && nWrites == 1 && nRWs == 0) ||
          !(memOp.getReadLatency() == 1 && memOp.getWriteLatency() == 1))
        return WalkResult::skip();
      SmallVector<Attribute, 4> resultNames;
      SmallVector<Type, 4> resultTypes;
      SmallVector<Attribute> portAtts;
      SmallVector<Attribute, 4> portAnnotations;
      Value rClock, wClock;
      // The memory has exactly two ports.
      SmallVector<Value> readTerms, writeTerms;
      for (const auto &portIt : llvm::enumerate(memOp.getResults())) {
        Attribute portAnno;
        portAnno = memOp.getPortAnnotation(portIt.index());
        if (memOp.getPortKind(portIt.index()) == MemOp::PortKind::Debug) {
          resultNames.push_back(memOp.getPortName(portIt.index()));
          resultTypes.push_back(memOp.getResult(portIt.index()).getType());
          portAnnotations.push_back(portAnno);
          continue;
        }
        // Append the annotations from the two ports.
        if (!cast<ArrayAttr>(portAnno).empty())
          portAtts.push_back(memOp.getPortAnnotation(portIt.index()));
        // Get the port value.
        Value portVal = portIt.value();
        // Get the port kind.
        bool isReadPort =
            memOp.getPortKind(portIt.index()) == MemOp::PortKind::Read;
        // Iterate over all users of the port.
        for (Operation *u : portVal.getUsers())
          if (auto sf = dyn_cast<SubfieldOp>(u)) {
            // Get the field name.
            auto fName = sf.getInput().getType().base().getElementName(
                sf.getFieldIndex());
            // If this is the enable field, record the product terms(the And
            // expression tree).
            if (fName == "en")
              getProductTerms(sf, isReadPort ? readTerms : writeTerms);

            else if (fName == "clk") {
              if (isReadPort)
                rClock = getConnectSrc(sf);
              else
                wClock = getConnectSrc(sf);
            }
          }
        // End of loop for getting MemOp port users.
      }
      if (!sameDriver(rClock, wClock))
        return WalkResult::skip();

      rClock = wClock;
      LLVM_DEBUG(
          llvm::dbgs() << "\n read clock:" << rClock
                       << " --- write clock:" << wClock;
          llvm::dbgs() << "\n Read terms==>"; for (auto t
                                                   : readTerms) llvm::dbgs()
                                              << "\n term::" << t;

          llvm::dbgs() << "\n Write terms==>"; for (auto t
                                                    : writeTerms) llvm::dbgs()
                                               << "\n term::" << t;

      );
      // If the read and write clocks are the same, and if any of the write
      // enable product terms are a complement of the read enable, then return
      // the write enable term.
      auto complementTerm = checkComplement(readTerms, writeTerms);
      if (!complementTerm)
        return WalkResult::skip();

      // Create the merged rw port for the new memory.
      resultNames.push_back(StringAttr::get(memOp.getContext(), "rw"));
      // Set the type of the rw port.
      resultTypes.push_back(MemOp::getTypeForPort(
          memOp.getDepth(), memOp.getDataType(), MemOp::PortKind::ReadWrite,
          memOp.getMaskBits()));
      ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
      portAnnotations.push_back(builder.getArrayAttr(portAtts));
      // Create the new rw memory.
      auto rwMem = MemOp::create(
          builder, resultTypes, memOp.getReadLatency(), memOp.getWriteLatency(),
          memOp.getDepth(), RUWAttr::Undefined,
          builder.getArrayAttr(resultNames), memOp.getNameAttr(),
          memOp.getNameKind(), memOp.getAnnotations(),
          builder.getArrayAttr(portAnnotations), memOp.getInnerSymAttr(),
          memOp.getInitAttr(), memOp.getPrefixAttr());
      ++numRWPortMemoriesInferred;
      auto rwPort = rwMem->getResult(nDbgs);
      // Create the subfield access to all fields of the port.
      // The addr should be connected to read/write address depending on the
      // read/write mode.
      auto addr = SubfieldOp::create(builder, rwPort, "addr");
      // Enable is high whenever the memory is written or read.
      auto enb = SubfieldOp::create(builder, rwPort, "en");
      // Read/Write clock.
      auto clk = SubfieldOp::create(builder, rwPort, "clk");
      auto readData = SubfieldOp::create(builder, rwPort, "rdata");
      // wmode is high when the port is in write mode. That is this can be
      // connected to the write enable.
      auto wmode = SubfieldOp::create(builder, rwPort, "wmode");
      auto writeData = SubfieldOp::create(builder, rwPort, "wdata");
      auto mask = SubfieldOp::create(builder, rwPort, "wmask");
      // Temp wires to replace the original memory connects.
      auto rAddr =
          WireOp::create(builder, addr.getType(), "readAddr").getResult();
      auto wAddr =
          WireOp::create(builder, addr.getType(), "writeAddr").getResult();
      auto wEnWire =
          WireOp::create(builder, enb.getType(), "writeEnable").getResult();
      auto rEnWire =
          WireOp::create(builder, enb.getType(), "readEnable").getResult();
      auto writeClock =
          WireOp::create(builder, ClockType::get(enb.getContext())).getResult();
      // addr = Mux(WriteEnable, WriteAddress, ReadAddress).
      MatchingConnectOp::create(
          builder, addr, MuxPrimOp::create(builder, wEnWire, wAddr, rAddr));
      // Enable = Or(WriteEnable, ReadEnable).
      MatchingConnectOp::create(builder, enb,
                                OrPrimOp::create(builder, rEnWire, wEnWire));
      builder.setInsertionPointToEnd(wmode->getBlock());
      MatchingConnectOp::create(builder, wmode, complementTerm);
      // Now iterate over the original memory read and write ports.
      size_t dbgsIndex = 0;
      for (const auto &portIt : llvm::enumerate(memOp.getResults())) {
        // Get the port value.
        Value portVal = portIt.value();
        if (memOp.getPortKind(portIt.index()) == MemOp::PortKind::Debug) {
          memOp.getResult(portIt.index())
              .replaceAllUsesWith(rwMem.getResult(dbgsIndex));
          dbgsIndex++;
          continue;
        }
        // Get the port kind.
        bool isReadPort =
            memOp.getPortKind(portIt.index()) == MemOp::PortKind::Read;
        // Iterate over all users of the port, which are the subfield ops, and
        // replace them.
        for (Operation *u : portVal.getUsers())
          if (auto sf = dyn_cast<SubfieldOp>(u)) {
            StringRef fName = sf.getInput().getType().base().getElementName(
                sf.getFieldIndex());
            Value repl;
            if (isReadPort)
              repl = llvm::StringSwitch<Value>(fName)
                         .Case("en", rEnWire)
                         .Case("clk", clk)
                         .Case("addr", rAddr)
                         .Case("data", readData);
            else
              repl = llvm::StringSwitch<Value>(fName)
                         .Case("en", wEnWire)
                         .Case("clk", writeClock)
                         .Case("addr", wAddr)
                         .Case("data", writeData)
                         .Case("mask", mask);
            sf.replaceAllUsesWith(repl);
            // Once all the uses of the subfield op replaced, delete it.
            opsToErase.push_back(sf);
          }
      }
      simplifyWmode(rwMem);
      // All uses for all results of mem removed, now erase the memOp.
      opsToErase.push_back(memOp);
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return signalPassFailure();

    for (auto *o : opsToErase)
      o->erase();
  }

private:
  // Get the source value which is connected to the dst.
  Value getConnectSrc(Value dst) {
    for (auto *c : dst.getUsers())
      if (auto connect = dyn_cast<FConnectLike>(c))
        if (connect.getDest() == dst)
          return connect.getSrc();

    return nullptr;
  }

  /// If the ports are not directly connected to the same clock, then check
  /// if indirectly connected to the same clock.
  bool sameDriver(Value rClock, Value wClock) {
    if (rClock == wClock)
      return true;
    DenseSet<Value> rClocks, wClocks;
    while (rClock) {
      // Record all the values which are indirectly connected to the clock
      // port.
      rClocks.insert(rClock);
      rClock = getConnectSrc(rClock);
    }

    bool sameClock = false;
    // Now check all the indirect connections to the write memory clock
    // port.
    while (wClock) {
      if (rClocks.find(wClock) != rClocks.end()) {
        sameClock = true;
        break;
      }
      wClock = getConnectSrc(wClock);
    }
    return sameClock;
  }

  void getProductTerms(Value enValue, SmallVector<Value> &terms) {
    if (!enValue)
      return;
    SmallVector<Value> worklist;
    worklist.push_back(enValue);
    while (!worklist.empty()) {
      auto term = worklist.back();
      worklist.pop_back();
      terms.push_back(term);
      if (isa<BlockArgument>(term))
        continue;
      TypeSwitch<Operation *>(term.getDefiningOp())
          .Case<NodeOp>([&](auto n) { worklist.push_back(n.getInput()); })
          .Case<AndPrimOp>([&](AndPrimOp andOp) {
            worklist.push_back(andOp.getOperand(0));
            worklist.push_back(andOp.getOperand(1));
          })
          .Case<MuxPrimOp>([&](auto muxOp) {
            // Check for the pattern when low is 0, which is equivalent to (sel
            // & high)
            // term = mux (sel, high, 0) => term = sel & high
            if (ConstantOp cLow = dyn_cast_or_null<ConstantOp>(
                    muxOp.getLow().getDefiningOp()))
              if (cLow.getValue().isZero()) {
                worklist.push_back(muxOp.getSel());
                worklist.push_back(muxOp.getHigh());
              }
          })
          .Default([&](auto) {
            if (auto src = getConnectSrc(term))
              worklist.push_back(src);
          });
    }
  }

  /// If any of the terms in the read enable, prodTerms[0] is a complement of
  /// any of the terms in the write enable prodTerms[1], return the
  /// corresponding write enable term. prodTerms[0], prodTerms[1] is a vector of
  /// Value, each of which correspond to the two product terms of read and write
  /// enable respectively.
  Value checkComplement(const SmallVector<Value> &readTerms,
                        const SmallVector<Value> &writeTerms) {
    // Foreach Value in first term, check if it is the complement of any of the
    // Value in second term.
    for (auto t1 : readTerms)
      for (auto t2 : writeTerms) {
        // Return t2, t1 is a Not of t2.
        if (!isa<BlockArgument>(t1) && isa<NotPrimOp>(t1.getDefiningOp()))
          if (cast<NotPrimOp>(t1.getDefiningOp()).getInput() == t2)
            return t2;
        // Else Return t2, if t2 is a Not of t1.
        if (!isa<BlockArgument>(t2) && isa<NotPrimOp>(t2.getDefiningOp()))
          if (cast<NotPrimOp>(t2.getDefiningOp()).getInput() == t1)
            return t2;
      }

    return {};
  }

  void handleCatPrimOp(CatPrimOp defOp, SmallVectorImpl<Value> &bits) {

    long lastSize = 0;
    // Cat the bits of both the operands.
    for (auto operand : defOp->getOperands()) {
      SmallVectorImpl<Value> &opBits = valueBitsSrc[operand];
      size_t s =
          getBitWidth(type_cast<FIRRTLBaseType>(operand.getType())).value();
      assert(opBits.size() == s);
      for (long i = lastSize, e = lastSize + s; i != e; ++i)
        bits[i] = opBits[i - lastSize];
      lastSize = s;
    }
  }

  void handleBitsPrimOp(BitsPrimOp bitsPrim, SmallVectorImpl<Value> &bits) {

    SmallVectorImpl<Value> &opBits = valueBitsSrc[bitsPrim.getInput()];
    for (size_t srcIndex = bitsPrim.getLo(), e = bitsPrim.getHi(), i = 0;
         srcIndex <= e; ++srcIndex, ++i)
      bits[i] = opBits[srcIndex];
  }

  // Try to extract the value assigned to each bit of `val`. This is a heuristic
  // to determine if each bit of the `val` is assigned the same value.
  // Common pattern that this heuristic detects,
  // mask = {{w1,w1},{w2,w2}}}
  // w1 = w[0]
  // w2 = w[0]
  bool areBitsDrivenBySameSource(Value val) {
    SmallVector<Value> stack;
    stack.push_back(val);

    while (!stack.empty()) {
      auto val = stack.back();
      if (valueBitsSrc.contains(val)) {
        stack.pop_back();
        continue;
      }

      auto size = getBitWidth(type_cast<FIRRTLBaseType>(val.getType()));
      // Cannot analyze aggregate types.
      if (!size.has_value())
        return false;

      auto bitsSize = size.value();
      if (auto *defOp = val.getDefiningOp()) {
        if (isa<CatPrimOp>(defOp)) {
          bool operandsDone = true;
          // If the value is a cat of other values, compute the bits of the
          // operands.
          for (auto operand : defOp->getOperands()) {
            if (valueBitsSrc.contains(operand))
              continue;
            stack.push_back(operand);
            operandsDone = false;
          }
          if (!operandsDone)
            continue;

          valueBitsSrc[val].resize_for_overwrite(bitsSize);
          handleCatPrimOp(cast<CatPrimOp>(defOp), valueBitsSrc[val]);
        } else if (auto bitsPrim = dyn_cast<BitsPrimOp>(defOp)) {
          auto input = bitsPrim.getInput();
          if (!valueBitsSrc.contains(input)) {
            stack.push_back(input);
            continue;
          }
          valueBitsSrc[val].resize_for_overwrite(bitsSize);
          handleBitsPrimOp(bitsPrim, valueBitsSrc[val]);
        } else if (auto constOp = dyn_cast<ConstantOp>(defOp)) {
          auto constVal = constOp.getValue();
          valueBitsSrc[val].resize_for_overwrite(bitsSize);
          if (constVal.isAllOnes() || constVal.isZero()) {
            for (auto &b : valueBitsSrc[val])
              b = constOp;
          } else
            return false;
        } else if (auto wireOp = dyn_cast<WireOp>(defOp)) {
          if (bitsSize != 1)
            return false;
          valueBitsSrc[val].resize_for_overwrite(bitsSize);
          if (auto src = getConnectSrc(wireOp.getResult())) {
            valueBitsSrc[val][0] = src;
          } else
            valueBitsSrc[val][0] = wireOp.getResult();
        } else
          return false;
      } else
        return false;
      stack.pop_back();
    }
    if (!valueBitsSrc.contains(val))
      return false;
    return llvm::all_equal(valueBitsSrc[val]);
  }

  // Remove redundant dependence of wmode on the enable signal. wmode can assume
  // the enable signal be true.
  void simplifyWmode(MemOp &memOp) {

    // Iterate over all results, and find the enable and wmode fields of the RW
    // port.
    for (const auto &portIt : llvm::enumerate(memOp.getResults())) {
      auto portKind = memOp.getPortKind(portIt.index());
      if (portKind != MemOp::PortKind::ReadWrite)
        continue;
      Value enableDriver, wmodeDriver;
      Value portVal = portIt.value();
      // Iterate over all users of the rw port.
      for (Operation *u : portVal.getUsers())
        if (auto sf = dyn_cast<SubfieldOp>(u)) {
          // Get the field name.
          auto fName =
              sf.getInput().getType().base().getElementName(sf.getFieldIndex());
          // Record the enable and wmode fields.
          if (fName.contains("en"))
            enableDriver = getConnectSrc(sf.getResult());
          if (fName.contains("wmode"))
            wmodeDriver = getConnectSrc(sf.getResult());
        }

      if (enableDriver && wmodeDriver) {
        ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
        builder.setInsertionPointToStart(
            memOp->getParentOfType<FModuleOp>().getBodyBlock());
        auto constOne = ConstantOp::create(
            builder, UIntType::get(builder.getContext(), 1), APInt(1, 1));
        setEnable(enableDriver, wmodeDriver, constOne);
      }
    }
  }

  // Replace any occurence of enable on the expression tree of wmode with a
  // constant one.
  void setEnable(Value enableDriver, Value wmodeDriver, Value constOne) {
    auto getDriverOp = [&](Value dst) -> Operation * {
      // Look through one level of wire to get the driver op.
      auto *defOp = dst.getDefiningOp();
      if (defOp) {
        if (isa<WireOp>(defOp))
          dst = getConnectSrc(dst);
        if (dst)
          defOp = dst.getDefiningOp();
      }
      return defOp;
    };
    SmallVector<Value> stack;
    llvm::SmallDenseSet<Value> visited;
    stack.push_back(wmodeDriver);
    while (!stack.empty()) {
      auto driver = stack.pop_back_val();
      if (!visited.insert(driver).second)
        continue;
      auto *defOp = getDriverOp(driver);
      if (!defOp)
        continue;
      for (auto operand : llvm::enumerate(defOp->getOperands())) {
        if (operand.value() == enableDriver)
          defOp->setOperand(operand.index(), constOne);
        else
          stack.push_back(operand.value());
      }
    }
  }

  void inferUnmasked(MemOp &memOp, SmallVector<Operation *> &opsToErase) {
    bool isMasked = true;

    // Iterate over all results, and check if the mask field of the result is
    // connected to a multi-bit constant 1.
    for (const auto &portIt : llvm::enumerate(memOp.getResults())) {
      // Read ports donot have the mask field.
      if (memOp.getPortKind(portIt.index()) == MemOp::PortKind::Read ||
          memOp.getPortKind(portIt.index()) == MemOp::PortKind::Debug)
        continue;
      Value portVal = portIt.value();
      // Iterate over all users of the write/rw port.
      for (Operation *u : portVal.getUsers())
        if (auto sf = dyn_cast<SubfieldOp>(u)) {
          // Get the field name.
          auto fName =
              sf.getInput().getType().base().getElementName(sf.getFieldIndex());
          // Check if this is the mask field.
          if (fName.contains("mask")) {
            // Already 1 bit, nothing to do.
            if (sf.getResult().getType().getBitWidthOrSentinel() == 1)
              continue;
            // Check what is the mask field directly connected to.
            // If we can infer that all the bits of the mask are always assigned
            // the same value, then the memory is unmasked.
            if (auto maskVal = getConnectSrc(sf))
              if (areBitsDrivenBySameSource(maskVal))
                isMasked = false;
          }
        }
    }

    if (!isMasked) {
      // Replace with a new memory of 1 bit mask.
      ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
      // Copy the result type, except the mask bits.
      SmallVector<Type, 4> resultTypes;
      for (size_t i = 0, e = memOp.getNumResults(); i != e; ++i)
        resultTypes.push_back(
            MemOp::getTypeForPort(memOp.getDepth(), memOp.getDataType(),
                                  memOp.getPortKind(i), /*maskBits=*/1));

      // Copy everything from old memory, except the result type.
      auto newMem = MemOp::create(
          builder, resultTypes, memOp.getReadLatencyAttr(),
          memOp.getWriteLatencyAttr(), memOp.getDepthAttr(), memOp.getRuwAttr(),
          memOp.getPortNamesAttr(), memOp.getNameAttr(),
          memOp.getNameKindAttr(), memOp.getAnnotationsAttr(),
          memOp.getPortAnnotationsAttr(), memOp.getInnerSymAttr(),
          memOp.getInitAttr(), memOp.getPrefixAttr());
      // Now replace the result of old memory with the new one.
      for (const auto &portIt : llvm::enumerate(memOp.getResults())) {
        // Old result.
        Value oldPort = portIt.value();
        // New result.
        auto newPortVal = newMem->getResult(portIt.index());
        // If read port, then blindly replace.
        if (memOp.getPortKind(portIt.index()) == MemOp::PortKind::Read ||
            memOp.getPortKind(portIt.index()) == MemOp::PortKind::Debug) {
          oldPort.replaceAllUsesWith(newPortVal);
          continue;
        }
        // Otherwise, all fields can be blindly replaced, except mask field.
        for (Operation *u : oldPort.getUsers()) {
          auto oldRes = dyn_cast<SubfieldOp>(u);
          auto sf =
              SubfieldOp::create(builder, newPortVal, oldRes.getFieldIndex());
          auto fName =
              sf.getInput().getType().base().getElementName(sf.getFieldIndex());
          // Replace all mask fields with a one bit constant 1.
          // Replace all other fields with the new port.
          if (fName.contains("mask")) {
            WireOp dummy = WireOp::create(builder, oldRes.getType());
            oldRes->replaceAllUsesWith(dummy);
            MatchingConnectOp::create(
                builder, sf,
                ConstantOp::create(builder,
                                   UIntType::get(builder.getContext(), 1),
                                   APInt(1, 1)));
          } else
            oldRes->replaceAllUsesWith(sf);

          opsToErase.push_back(oldRes);
        }
      }
      opsToErase.push_back(memOp);
      memOp = newMem;
    }
  }

  // Record of what are the source values that drive each bit of a value. Used
  // to check if each bit of a value is being driven by the same source.
  llvm::DenseMap<Value, SmallVector<Value>> valueBitsSrc;
};
} // end anonymous namespace
