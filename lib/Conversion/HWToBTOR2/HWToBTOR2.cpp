//===- HWToBTOR2.cpp - HW to BTOR2 translation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Converts a hw module to a btor2 format and prints it out
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToBTOR2.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "circt/Dialect/SV/SVVisitors.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace hw;

namespace {
// The goal here is to traverse the operations in order and convert them one by
// one into btor2
struct ConvertHWToBTOR2Pass
    : public ConvertHWToBTOR2Base<ConvertHWToBTOR2Pass>,
      public comb::CombinationalVisitor<ConvertHWToBTOR2Pass>,
      public sv::Visitor<ConvertHWToBTOR2Pass>,
      public hw::TypeOpVisitor<ConvertHWToBTOR2Pass> {
public:
  ConvertHWToBTOR2Pass(raw_ostream &os) : os(os) {}
  // Executes the pass
  void runOnOperation() override;

private:
  // Output stream in which the btor2 will be emitted
  raw_ostream &os;

  // Create a counter that attributes a unique id to each generated btor2 line
  size_t lid = 1;          // btor2 line identifiers usually start at 1
  size_t resetLID = noLID; // keeps track of the reset's LID
  size_t nclocks = 0;

  // Create maps to keep track of lid associations
  // We need these in order to reference results as operands in btor2

  // Keeps track of the ids associated to each declared sort
  // This is used in order to guarantee that sorts are unique and to allow for
  // instructions to reference the given sorts (key: width, value: LID)
  DenseMap<size_t, size_t> sortToLIDMap;
  // Keeps track of {constant, width} -> LID mappings
  // This is used in order to avoid duplicating constant declarations
  // in the output btor2. It is also useful when tracking
  // constants declarations that aren't tied to MLIR ops.
  DenseMap<APInt, size_t> constToLIDMap;
  // Keeps track of the most recent update line for each operation
  // This allows for operations to be used throughout the btor file
  // with their most recent expression. Btor uses unique identifiers for each
  // instruction, so we need to have an association between those and MLIR Ops.
  DenseMap<Operation *, size_t> opLIDMap;
  // Keeps track of operation aliases. This is used for wire inlining, as
  // btor2 does not have the concept of a wire. This means that wires in
  // hw will simply create an alias for the operation that will point to
  // the same LID as the original op.
  // key: alias, value: original op
  DenseMap<Operation *, Operation *> opAliasMap;
  // Stores the LID of the associated input.
  // This holds a similar function as the opLIDMap but keeps
  // track of block argument index -> LID mappings
  DenseMap<size_t, size_t> inputLIDs;
  // Stores all of the register declaration ops.
  // This allows for the emission of transition arcs for the regs
  // to be deferred to the end of the pass.
  // This is necessary, as we need to wait for the `next` operation to
  // have been converted to btor2 before we can emit the transition.
  SmallVector<Operation *> regOps;

  // Used to perform a DFS search through the module to declare all operands
  // before they are used
  llvm::SmallMapVector<Operation *, OperandRange::iterator, 16> worklist;

  // Keeps track of operations that have been declared
  DenseSet<Operation *> handledOps;

  // Constants used during the conversion
  static constexpr size_t noLID = -1UL;
  static constexpr int64_t noWidth = -1L;

  /// Field helper functions
public:
  // Checks if an operation was declared
  // If so, its lid will be returned
  // Otherwise a new lid will be assigned to the op
  size_t getOpLID(Operation *op) {
    // Look for the original operation declaration
    // Make sure that wires are considered when looking for an lid
    Operation *defOp = getOpAlias(op);
    auto &f = opLIDMap[defOp];

    // If the op isn't associated to an lid, assign it a new one
    if (!f)
      f = lid++;
    return f;
  }

  // Associates the current lid to an operation
  // The LID is then incremented to maintain uniqueness
  size_t setOpLID(Operation *op) {
    size_t oplid = lid++;
    opLIDMap[op] = oplid;
    return oplid;
  }

  // Checks if an operation was declared
  // If so, its lid will be returned
  // Otherwise -1 will be returned
  size_t getOpLID(Value value) {
    // Check for an operation alias
    Operation *defOp = getOpAlias(value.getDefiningOp());

    if (auto it = opLIDMap.find(defOp); it != opLIDMap.end())
      return it->second;

    // Check for special case where op is actually a port
    // To do so, we start by checking if our operation is a block argument
    if (BlockArgument barg = dyn_cast<BlockArgument>(value)) {
      // Extract the block argument index and use that to get the line number
      size_t argIdx = barg.getArgNumber();

      // Check that the extracted argument is in range before using it
      if (auto it = inputLIDs.find(argIdx); it != inputLIDs.end())
        return it->second;
    }

    // Return -1 if no LID was found
    return noLID;
  }

private:
  // Checks if an operation has an alias. This is the case for wires
  // If so, the original operation is returned
  // Otherwise the argument is returned as it is the original op
  Operation *getOpAlias(Operation *op) {
    if (auto it = opAliasMap.find(op); it != opAliasMap.end())
      return it->second;
    // If the op isn't an alias then simply return it
    return op;
  }

  // Updates or creates an entry for the given operation
  // associating it with the current lid
  void setOpAlias(Operation *alias, Operation *op) { opAliasMap[alias] = op; }

  // Checks if a sort was declared with the given width
  // If so, its lid will be returned
  // Otherwise -1 will be returned
  size_t getSortLID(size_t w) {
    if (auto it = sortToLIDMap.find(w); it != sortToLIDMap.end())
      return it->second;

    // If no lid was found return -1
    return noLID;
  }

  // Associate the sort with a new lid
  size_t setSortLID(size_t w) {
    size_t sortlid = lid;
    // Add the width to the declared sorts along with the associated line id
    sortToLIDMap[w] = lid++;
    return sortlid;
  }

  // Checks if a constant of a given size has been declared.
  // If so, its lid will be returned.
  // Otherwise -1 will be returned.
  size_t getConstLID(int64_t val, size_t w) {
    if (auto it = constToLIDMap.find(APInt(w, val)); it != constToLIDMap.end())
      return it->second;

    // if no lid was found return -1
    return noLID;
  }

  // Associates a constant declaration to a new lid
  size_t setConstLID(int64_t val, size_t w) {
    size_t constlid = lid;
    // Keep track of this value in a constant declaration tracker
    constToLIDMap[APInt(w, val)] = lid++;
    return constlid;
  }

  /// String generation helper functions

  // Generates a sort declaration instruction given a type ("bitvec" or array)
  // and a width.
  void genSort(StringRef type, size_t width) {
    // Check that the sort wasn't already declared
    if (getSortLID(width) != noLID) {
      return; // If it has already been declared then return an empty string
    }

    size_t sortlid = setSortLID(width);

    // Build and return a sort declaration
    os << sortlid << " "
       << "sort"
       << " " << type << " " << width << "\n";
  }

  // Generates an input declaration given a sort lid and a name.
  void genInput(size_t inlid, size_t width, StringRef name) {
    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Generate input declaration
    os << inlid << " "
       << "input"
       << " " << sid << " " << name << "\n";
  }

  // Generates a constant declaration given a value, a width and a name.
  void genConst(int64_t value, size_t width, Operation *op) {
    // For now we're going to assume that the name isn't taken, given that hw is
    // already in SSA form
    size_t opLID = getOpLID(op);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    os << opLID << " "
       << "constd"
       << " " << sid << " " << value << "\n";
  }

  // Generates a zero constant expression
  size_t genZero(size_t width) {
    // Check if the constant has been created yet
    size_t zlid = getConstLID(0, width);
    if (zlid != noLID)
      return zlid;

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Associate an lid to the new constant
    size_t constlid = setConstLID(0, width);

    // Build and return the zero btor instruction
    os << constlid << " "
       << "zero"
       << " " << sid << "\n";
    return constlid;
  }

  // Generates a binary operation instruction given an op name, two operands and
  // a result width.
  void genBinOp(StringRef inst, Operation *binop, Value op1, Value op2,
                size_t width) {
    // Set the LID for this operation
    size_t opLID = getOpLID(binop);

    // Find the sort's lid
    size_t sid = sortToLIDMap.at(width);

    // Assuming that the operands were already emitted
    // Find the LIDs associated to the operands
    size_t op1LID = getOpLID(op1);
    size_t op2LID = getOpLID(op2);

    // Build and return the string
    os << opLID << " " << inst << " " << sid << " " << op1LID << " " << op2LID
       << "\n";
  }

  // Generates a slice instruction given an operand, the lowbit, and the width
  void genSlice(Operation *srcop, Value op0, size_t lowbit, int64_t width) {
    // Assign a LID to this operation
    size_t opLID = getOpLID(srcop);

    // Find the sort's associated lid in order to use it in the instruction
    size_t sid = sortToLIDMap.at(width);

    // Assuming that the operand has already been emitted
    // Find the LID associated to the operand
    size_t op0LID = getOpLID(op0);

    // Build and return the slice instruction
    os << opLID << " "
       << "slice"
       << " " << sid << " " << op0LID << " " << (width - 1) << " " << lowbit
       << "\n";
  }

  // Generates a constant declaration given a value, a width and a name
  void genUnaryOp(Operation *srcop, Operation *op0, StringRef inst,
                  size_t width) {
    // Register the source operation with the current line id
    size_t opLID = getOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Assuming that the operand has already been emitted
    // Find the LID associated to the operand
    size_t op0LID = getOpLID(op0);

    os << opLID << " " << inst << " " << sid << " " << op0LID << "\n";
  }

  // Generates a constant declaration given a value, a width and a name
  void genUnaryOp(Operation *srcop, Value op0, StringRef inst, size_t width) {
    genUnaryOp(srcop, op0.getDefiningOp(), inst, width);
  }

  // Generate a btor2 assertion given an assertion operation
  // Note that a predicate inversion must have already been generated at this
  // point
  void genBad(Operation *assertop) {
    // Start by finding the expression lid
    size_t assertLID = getOpLID(assertop);

    // Build and return the btor2 string
    // Also update the lid as this instruction is not associated to an mlir op
    os << lid++ << " "
       << "bad"
       << " " << assertLID << "\n";
  }

  // Generate a btor2 constraint given an expression from an assumption
  // operation
  void genConstraint(Value expr) {
    // Start by finding the expression lid
    size_t exprLID = getOpLID(expr);

    genConstraint(exprLID);
  }

  // Generate a btor2 constraint given an expression from an assumption
  // operation
  void genConstraint(size_t exprLID) {
    // Build and return the btor2 string
    // Also update the lid as this instruction is not associated to an mlir op
    os << lid++ << " "
       << "constraint"
       << " " << exprLID << "\n";
  }

  // Generate an ite instruction (if then else) given a predicate, two values
  // and a res width
  void genIte(Operation *srcop, Value cond, Value t, Value f, int64_t width) {
    // Retrieve the operand lids, assuming they were emitted
    size_t condLID = getOpLID(cond);
    size_t tLID = getOpLID(t);
    size_t fLID = getOpLID(f);

    genIte(srcop, condLID, tLID, fLID, width);
  }

  // Generate an ite instruction (if then else) given a predicate, two values
  // and a res width
  void genIte(Operation *srcop, size_t condLID, size_t tLID, size_t fLID,
              int64_t width) {
    // Register the source operation with the current line id
    size_t opLID = getOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Build and return the ite instruction
    os << opLID << " "
       << "ite"
       << " " << sid << " " << condLID << " " << tLID << " " << fLID << "\n";
  }

  // Generate a logical implication given a lhs and a rhs
  void genImplies(Operation *srcop, Value lhs, Value rhs) {
    // Retrieve LIDs for the lhs and rhs
    size_t lhsLID = getOpLID(lhs);
    size_t rhsLID = getOpLID(rhs);

    genImplies(srcop, lhsLID, rhsLID);
  }

  // Generate a logical implication given a lhs and a rhs
  void genImplies(Operation *srcop, size_t lhsLID, size_t rhsLID) {
    // Register the source operation with the current line id
    size_t opLID = getOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(1);

    // Build and emit the implies operation
    os << opLID << " "
       << "implies"
       << " " << sid << " " << lhsLID << " " << rhsLID << "\n";
  }

  // Generates a state instruction given a width and a name
  void genState(Operation *srcop, int64_t width, StringRef name) {
    // Register the source operation with the current line id
    size_t opLID = getOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Build and return the state instruction
    os << opLID << " "
       << "state"
       << " " << sid << " " << name << "\n";
  }

  // Generates a next instruction, given a width, a state LID, and a next value
  // LID
  void genNext(Value next, Operation *reg, int64_t width) {
    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Retrieve the LIDs associated to reg and next
    size_t regLID = getOpLID(reg);
    size_t nextLID = getOpLID(next);

    // Build and return the next instruction
    // Also update the lid as this instruction is not associated to an mlir op
    os << lid++ << " "
       << "next"
       << " " << sid << " " << regLID << " " << nextLID << "\n";
  }

  // Verifies that the sort required for the given operation's btor2 emission
  // has been generated
  int64_t requireSort(mlir::Type type) {
    // Start by figuring out what sort needs to be generated
    int64_t width = hw::getBitWidth(type);

    // Sanity check: getBitWidth can technically return -1 it is a type with no
    // width (like a clock). This shouldn't be allowed as width is required to
    // generate a sort
    assert(width != noWidth);

    // Generate the sort regardles of resulting width (nothing will be added if
    // the sort already exists)
    genSort("bitvec", width);
    return width;
  }

  // Generates the transitions required to finalize the register to state
  // transition system conversion
  void finalizeRegVisit(Operation *op) {
    // Check the register type (done to support non-firrtl registers as well)
    auto reg = cast<seq::FirRegOp>(op);

    // Generate the reset condition (for sync & async resets)
    // We assume for now that the reset value is always 0
    size_t width = hw::getBitWidth(reg.getType());
    genSort("bitvec", width);

    // Extract the `next` operation for each register (used to define the
    // transition). We need to check if next is a port to avoid nullptrs
    Value next = reg.getNext();

    // Next should already be associated to an LID at this point
    // As we are going to override it, we need to keep track of the original
    // instruction
    size_t nextLID = noLID;

    // Check for special case where next is actually a port
    // To do so, we start by checking if our operation is a block argument
    if (BlockArgument barg = dyn_cast<BlockArgument>(next)) {
      // Extract the block argument index and use that to get the line number
      size_t argIdx = barg.getArgNumber();

      // Check that the extracted argument is in range before using it
      nextLID = inputLIDs[argIdx];

    } else {
      nextLID = getOpLID(next);
    }

    // Assign a new LID to next
    setOpLID(next.getDefiningOp());

    // Sanity check: at this point the next operation should have had it's btor2
    // counterpart emitted if not then something terrible must have happened.
    assert(nextLID != noLID);

    // Check if the register has a reset
    if (resetLID != noLID) {
      size_t resetValLID = noLID;

      // Check for a reset value, if none exists assume it's zero
      if (auto resval = reg.getResetValue())
        resetValLID = getOpLID(resval.getDefiningOp());
      else
        resetValLID = genZero(width);

      // Generate the ite for the register update reset condition
      // i.e. reg <= reset ? 0 : next
      genIte(next.getDefiningOp(), resetLID, resetValLID, nextLID, width);
    }

    // Finally generate the next statement
    genNext(next, reg, width);
  }

public:
  /// Visitor Methods used later on for pattern matching

  // Visitor for the inputs of the module.
  // This will generate additional sorts and input declaration explicitly for
  // btor2 Note that outputs are ignored in btor2 as they do not contribute to
  // the final assertions
  void visit(hw::PortInfo &port) {
    // Separate the inputs from outputs and generate the first btor2 lines for
    // input declaration We only consider ports with an explicit bit-width (so
    // ignore clocks)
    if (port.isInput() && !port.type.isa<seq::ClockType>()) {
      // Generate the associated btor declaration for the inputs
      StringRef iName = port.getName();

      // Guarantees that a sort will exist for the generation of this port's
      // translation into btor2
      int64_t w = requireSort(port.type);

      // Save lid for later
      size_t inlid = lid;

      // Record the defining operation's line ID (the module itself in the case
      // of ports)
      inputLIDs[port.argNum] = lid;

      // We assume that the explicit name is always %reset for reset ports
      if (iName == "reset")
        resetLID = lid;

      // Increment the lid to keep it unique
      lid++;

      genInput(inlid, w, iName);
    }
  }

  // Emits the associated btor2 operation for a constant. Note that for
  // simplicity, we will only emit `constd` in order to avoid bit-string
  // conversions
  void visitTypeOp(hw::ConstantOp op) {
    // Make sure that a sort has been created for our operation
    int64_t w = requireSort(op.getType());

    // Prepare for for const generation by extracting the const value and
    // generting the btor2 string
    int64_t value = op.getValue().getSExtValue();
    genConst(value, w, op);
  }

  // Wires can generally be ignored in bto2, however we do need
  // to keep track of the new alias it creates
  void visit(hw::WireOp op) {
    // Retrieve the aliased operation
    Operation *defOp = op.getOperand().getDefiningOp();
    // Wires don't output anything so just record alias
    setOpAlias(op, defOp);
  }

  void visitTypeOp(Operation *op) { visitInvalidTypeOp(op); }

  // Handles non-hw operations
  void visitInvalidTypeOp(Operation *op) {
    // Try comb ops
    dispatchCombinationalVisitor(op);
  }

  // Binary operations are all emitted the same way, so we can group them into
  // a single method.
  void visitBinOp(Operation *op, StringRef inst, int64_t w) {

    // Start by extracting the operands
    Value op1 = op->getOperand(0);
    Value op2 = op->getOperand(1);

    // Generate the line
    genBinOp(inst, op, op1, op2, w);
  }

  // Visitors for the binary ops
  void visitComb(comb::AddOp op) {
    visitBinOp(op, "add", requireSort(op.getType()));
  }
  void visitComb(comb::SubOp op) {
    visitBinOp(op, "sub", requireSort(op.getType()));
  }
  void visitComb(comb::MulOp op) {
    visitBinOp(op, "mul", requireSort(op.getType()));
  }
  void visitComb(comb::DivSOp op) {
    visitBinOp(op, "sdiv", requireSort(op.getType()));
  }
  void visitComb(comb::DivUOp op) {
    visitBinOp(op, "udiv", requireSort(op.getType()));
  }
  void visitComb(comb::ModSOp op) {
    visitBinOp(op, "smod", requireSort(op.getType()));
  }
  void visitComb(comb::ShlOp op) {
    visitBinOp(op, "sll", requireSort(op.getType()));
  }
  void visitComb(comb::ShrUOp op) {
    visitBinOp(op, "srl", requireSort(op.getType()));
  }
  void visitComb(comb::ShrSOp op) {
    visitBinOp(op, "sra", requireSort(op.getType()));
  }
  void visitComb(comb::AndOp op) {
    visitBinOp(op, "and", requireSort(op.getType()));
  }
  void visitComb(comb::OrOp op) {
    visitBinOp(op, "or", requireSort(op.getType()));
  }
  void visitComb(comb::XorOp op) {
    visitBinOp(op, "xor", requireSort(op.getType()));
  }
  void visitComb(comb::ConcatOp op) {
    visitBinOp(op, "concat", requireSort(op.getType()));
  }

  // Extract ops translate to a slice operation in btor2 in a one-to-one manner
  void visitComb(comb::ExtractOp op) {
    int64_t w = requireSort(op.getType());

    // Start by extracting the necessary information for the emission (i.e.
    // operand, low bit, ...)
    Value op0 = op.getOperand();
    size_t lb = op.getLowBit();

    // Generate the slice instruction
    genSlice(op, op0, lb, w);
  }

  // Btor2 uses similar syntax as hw for its comparisons
  // So we simply need to emit the cmpop name and check for corner cases
  // where the namings differ.
  void visitComb(comb::ICmpOp op) {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    // Extract the predicate name (assuming that its a valid btor2
    // predicate)
    StringRef pred = stringifyICmpPredicate(op.getPredicate());

    // Check for special cases where hw doesn't align with btor syntax
    if (pred == "ne")
      pred = "neq";

    // Width of result is always 1 for comparison
    genSort("bitvec", 1);

    // With the special cases out of the way, the emission is the same as that
    // of a binary op
    genBinOp(pred, op, lhs, rhs, 1);
  }

  // Muxes generally convert to an ite statement
  void visitComb(comb::MuxOp op) {
    // Extract predicate, true and false values
    Value pred = op.getCond();
    Value tval = op.getTrueValue();
    Value fval = op.getFalseValue();

    // We assume that both tval and fval have the same width
    // This width should be the same as the output width
    int64_t w = requireSort(op.getType());

    // Generate the ite instruction
    genIte(op, pred, tval, fval, w);
  }

  void visitComb(Operation *op) { visitInvalidComb(op); }

  void visitInvalidComb(Operation *op) {
    // try sv ops
    dispatchSVVisitor(op);
  }

  // Assertions are negated then converted to a btor2 bad instruction
  void visitSV(sv::AssertOp op) {
    // Expression is what we will try to invert for our assertion
    Value expr = op.getExpression();

    // This sort is for assertion inversion and potential implies
    genSort("bitvec", 1);

    // Check for an overaching enable
    // In our case the sv.if operation will probably only be used when
    // conditioning an sv.assert on an enable signal. This means that
    // its condition is probably used to imply our assertion
    if (auto ifop = dyn_cast<sv::IfOp>(((Operation *)op)->getParentOp())) {
      Value en = ifop.getOperand();

      // Generate the implication
      genImplies(ifop, en, expr);

      // Generate the implies inversion
      genUnaryOp(op, ifop, "not", 1);
    } else {
      // Generate the expression inversion
      genUnaryOp(op, expr, "not", 1);
    }

    // Genrate the bad btor2 intruction
    genBad(op);
  }
  // Assumptions are converted to a btor2 constraint instruction
  void visitSV(sv::AssumeOp op) {
    // Extract the expression that we want our constraint to be about
    Value expr = op.getExpression();
    genConstraint(expr);
  }

  void visitSV(Operation *op) { visitInvalidSV(op); }

  // Once SV Ops are visited, we need to check for seq ops
  void visitInvalidSV(Operation *op) { visit(op); }

  // Seq operation visitor, that dispatches to other seq ops
  // Also handles all remaining operations that should be explicitly ignored
  void visit(Operation *op) {
    // Typeswitch is used here because other seq types will be supported
    // like all operations relating to memories and CompRegs
    TypeSwitch<Operation *, void>(op)
        .Case<seq::FirRegOp, hw::WireOp>([&](auto expr) { visit(expr); })
        .Default([&](auto expr) { visitUnsupportedOp(op); });
  }

  // Firrtl registers generate a state instruction
  // The final update is also used to generate a set of next btor
  // instructions
  void visit(seq::FirRegOp reg) {
    // Start by retrieving the register's name and width
    StringRef regName = reg.getName();
    int64_t w = requireSort(reg.getType());

    // Generate state instruction (represents the register declaration)
    genState(reg, w, regName);

    // Record the operation for future `next` instruction generation
    // This is required to model transitions between states (i.e. how a
    // register's value evolves over time)
    regOps.push_back(reg);
  }

  // Ignore all other explicitly mentionned operations
  // ** Purposefully left empty **
  void ignore(Operation *op) {}

  // Tail method that handles all operations that weren't handled by previous
  // visitors. Here we simply make the pass fail or ignore the op
  void visitUnsupportedOp(Operation *op) {
    // Check for explicitly ignored ops vs unsupported ops (which cause a
    // failure)
    TypeSwitch<Operation *, void>(op)
        // All explicitly ignored operations are defined here
        .Case<sv::MacroDefOp, sv::MacroDeclOp, sv::VerbatimOp,
              sv::VerbatimExprOp, sv::VerbatimExprSEOp, sv::IfOp, sv::IfDefOp,
              sv::IfDefProceduralOp, sv::AlwaysOp, sv::AlwaysCombOp,
              sv::AlwaysFFOp, seq::FromClockOp, hw::OutputOp, hw::HWModuleOp>(
            [&](auto expr) { ignore(op); })

        // Make sure that the design only contains one clock
        .Case<seq::FromClockOp>([&](auto expr) {
          if (++nclocks > 1UL) {
            op->emitOpError("Mutli-clock designs are not supported!");
            return signalPassFailure();
          }
        })

        // Anything else is considered unsupported and might cause a wrong
        // behavior if ignored, so an error is thrown
        .Default([&](auto expr) {
          op->emitOpError("is an unsupported operation");
          return signalPassFailure();
        });
  }
};
} // end anonymous namespace

void ConvertHWToBTOR2Pass::runOnOperation() {
  // Btor2 does not have the concept of modules or module
  // hierarchies, so we assume that no nested modules exist at this point.
  // This greatly simplifies translation.
  getOperation().walk([&](hw::HWModuleOp module) {
    // Start by extracting the inputs and generating appropriate instructions
    for (auto &port : module.getPortList()) {
      visit(port);
    }

    // Previsit all registers in the module in order to avoid dependency cylcles
    module.walk([&](Operation *op) {
      if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
        visit(reg);
        handledOps.insert(op);
      }
    });

    // Visit all of the operations in our module
    module.walk([&](Operation *op) {
      // Check: instances are not (yet) supported
      if (isa<hw::InstanceOp>(op)) {
        op->emitOpError("not supported in BTOR2 conversion");
        return;
      }

      // Don't process ops that have already been emitted
      if (handledOps.contains(op))
        return;

      // Fill in our worklist
      worklist.insert({op, op->operand_begin()});

      // Process the elements in our worklist
      while (!worklist.empty()) {
        auto &[op, operandIt] = worklist.back();
        if (operandIt == op->operand_end()) {
          // All of the operands have been emitted, it is safe to emit our op
          dispatchTypeOpVisitor(op);

          // Record that our op has been emitted
          handledOps.insert(op);
          worklist.pop_back();
          continue;
        }

        // Send the operands of our op to the worklist in case they are still
        // un-emitted
        Value operand = *(operandIt++);
        auto *defOp = operand.getDefiningOp();

        // Make sure that we don't emit the same operand twice
        if (!defOp || handledOps.contains(defOp))
          continue;

        // This is triggered if our operand is already in the worklist and
        // wasn't handled
        if (!worklist.insert({defOp, defOp->operand_begin()}).second) {
          op->emitError("dependency cycle");
          return;
        }
      }
    });

    // Iterate through the registers and generate the `next` instructions
    for (size_t i = 0; i < regOps.size(); ++i) {
      finalizeRegVisit(regOps[i]);
    }
  });
  // Clear data structures to allow for pass reuse
  sortToLIDMap.clear();
  constToLIDMap.clear();
  opLIDMap.clear();
  opAliasMap.clear();
  inputLIDs.clear();
  regOps.clear();
  handledOps.clear();
  worklist.clear();
}

// Constructor with a custom ostream
std::unique_ptr<mlir::Pass>
circt::createConvertHWToBTOR2Pass(llvm::raw_ostream &os) {
  return std::make_unique<ConvertHWToBTOR2Pass>(os);
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createConvertHWToBTOR2Pass() {
  return std::make_unique<ConvertHWToBTOR2Pass>(llvm::outs());
}
