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

  // Create maps to keep track of lid associations
  // We need these in order to reference resluts as operands in btor2

  // Keeps track of the ids associated to each declared sort
  // This is used in oder to guarantee that sorts are unique and to allow for
  // instructions to reference the given sorts (key: width, value: LID)
  DenseMap<size_t, size_t> sortToLIDMap;
  // Keeps track of {constant, width} -> LID mappings
  // This is used in order to avoid duplicating constant declarations
  // in the output btor2. It is also useful when tracking
  // constants declarations that aren't tied to MLIR ops.
  DenseMap<std::pair<int64_t, size_t>, size_t> constToLIDMap;
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

  // Set of often reused strings in btor2 emission (to avoid typos and enable
  // auto-complete)
  static constexpr StringLiteral sortStr = "sort";
  static constexpr StringLiteral bitvecStr = "bitvec";
  static constexpr StringLiteral inputStr = "input";
  static constexpr StringLiteral resetStr = "reset";
  static constexpr StringLiteral outputStr = "output";
  static constexpr StringLiteral zeroStr = "zero";
  static constexpr StringLiteral oneStr = "one";
  static constexpr StringLiteral constStr = "const";
  static constexpr StringLiteral constdStr = "constd";
  static constexpr StringLiteral consthStr = "consth";
  static constexpr StringLiteral sliceStr = "slice";
  static constexpr StringLiteral uextStr = "uext";
  static constexpr StringLiteral addStr = "add";
  static constexpr StringLiteral subStr = "sub";
  static constexpr StringLiteral mulStr = "mul";
  static constexpr StringLiteral andStr = "and";
  static constexpr StringLiteral orStr = "or";
  static constexpr StringLiteral xorStr = "xor";
  static constexpr StringLiteral sllStr = "sll";
  static constexpr StringLiteral srlStr = "srl"; // a.k.a. unsigned right shift
  static constexpr StringLiteral sraStr = "sra"; // a.k.a. signed right shift
  static constexpr StringLiteral sdivStr = "sdiv";
  static constexpr StringLiteral udivStr = "udiv";
  static constexpr StringLiteral smodStr = "smod";
  static constexpr StringLiteral concatStr = "concat";
  static constexpr StringLiteral notStr = "not";
  static constexpr StringLiteral neqStr = "neq";
  static constexpr StringLiteral hwNeqStr = "ne";
  static constexpr StringLiteral iteStr = "ite";
  static constexpr StringLiteral impliesStr = "implies"; // logical implication
  static constexpr StringLiteral stateStr = "state";     // Register state
  static constexpr StringLiteral nextStr = "next"; // Register state transition
  static constexpr StringLiteral badStr = "bad";
  static constexpr StringLiteral constraintStr = "constraint";
  static constexpr StringLiteral wsStr = " ";  // WhiteSpace
  static constexpr StringLiteral nlStr = "\n"; // NewLine

  // Constants used during the conversion
  static constexpr size_t noLID = -1UL;
  static constexpr int64_t noWidth = -1L;

  /// Field helper functions

  // Checks if an operation was declared
  // If so, its lid will be returned
  // Otherwise -1 will be returned
  size_t getOrCreateOpLID(Operation *op) {
    if (op == nullptr)
      return noLID;

    // Look for the original operation declaration
    // Make sure that wires are considered when looking for an lid
    Operation *defOp = getOpAlias(op);
    if (opLIDMap.contains(defOp)) {
      return opLIDMap[defOp];
    }

    // Create a new entry with the current LID
    // if the op was not yet registered
    opLIDMap[defOp] = lid;

    return opLIDMap.at(defOp);
  }

  // Checks if an operation was declared
  // If so, its lid will be returned
  // Otherwise -1 will be returned
  size_t getOrCreateOpLID(Value op) {
    // Check for an operation alias
    Operation *defOp = getOpAlias(op.getDefiningOp());

    if (opLIDMap.contains(defOp)) {
      return opLIDMap[defOp];
    }

    // Check for special case where op is actually a port
    // To do so, we start by checking if our operation is a block argument
    if (BlockArgument barg = dyn_cast<BlockArgument>(op)) {
      // Extract the block argument index and use that to get the line number
      size_t argIdx = barg.getArgNumber();

      // Check that the extracted argument is in range before using it
      if (inputLIDs.contains(argIdx)) {
        return inputLIDs[argIdx];
      }
    }

    // Create a new entry with the current LID
    // if the op was not yet registered
    opLIDMap[defOp] = lid;

    return opLIDMap.at(defOp);
  }

  // Checks if an operation has an alias. This is the case for wires
  // If so, the original operation is returned
  // Otherwise the argument is returned as it is the original op
  Operation *getOpAlias(Operation *op) {
    if (opAliasMap.contains(op)) {
      return opAliasMap[op];
    }
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
    if (sortToLIDMap.contains(w)) {
      return sortToLIDMap[w];
    }
    // If no lid was found return -1
    return noLID;
  }

  // Checks if a constant of a given size has been declared
  // If so, its lid will be returned
  // Otherwise -1 will be returned
  size_t getConstLID(int64_t val, size_t w) {
    if (constToLIDMap.contains({val, w})) {
      return constToLIDMap[{val, w}];
    }
    // if no lid was found return -1
    return noLID;
  }

  /// String generation helper functions

  // Generates a sort declaration instruction given a type (bitvecStr or array)
  // and a width
  void genSort(StringRef type, size_t width) {
    // Check that the sort wasn't already declared
    if (getSortLID(width) != noLID) {
      return; // If it has already been declared then return an empty string
    }

    // Add the width to the declared sorts along with the associated line id
    sortToLIDMap[width] = lid;

    // Build and return a sort declaration
    os << lid++ << wsStr << sortStr << wsStr << type << wsStr << width << nlStr;
  }

  // Generates an input declaration given a sort lid and a name
  void genInput(size_t width, StringRef name) {
    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Generate input declaration
    os << lid++ << wsStr << inputStr << wsStr << sid << wsStr << name << nlStr;
  }

  // Generates a constant declaration given a value, a width and a name
  void genConst(int64_t value, size_t width, Operation *op) {
    // For now we're going to assume that the name isn't taken, given that hw is
    // already in SSA form
    getOrCreateOpLID(op);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    os << lid++ << wsStr << constdStr << wsStr << sid << wsStr << value
       << nlStr;
  }

  // Generates a zero constant expression
  void genZero(size_t width) {
    // Check if the constant has been created yet
    if (getConstLID(0, width) != noLID) {
      return;
    }

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Keep track of this value in a constant declaration tracker
    constToLIDMap[{0, width}] = lid;

    // Build and return the zero btor instruction
    os << lid++ << wsStr << zeroStr << wsStr << sid << nlStr;
  }

  // Generates a binary operation instruction given an op name, two operands and
  // a result width
  void genBinOp(StringRef inst, Operation *binop, Value op1, Value op2,
                size_t width) {
    // Set the LID for this operation
    getOrCreateOpLID(binop);

    // Find the sort's lid
    size_t sid = sortToLIDMap.at(width);

    // Assuming that the operands were already emitted
    // Find the LIDs associated to the operands
    size_t op1LID = getOrCreateOpLID(op1);
    size_t op2LID = getOrCreateOpLID(op2);

    // Build and return the string
    os << lid++ << wsStr << inst << wsStr << sid << wsStr << op1LID << wsStr
       << op2LID << nlStr;
  }

  // Generates a slice instruction given an operand, the lowbit, and the width
  void genSlice(Operation *srcop, Value op0, size_t lowbit, int64_t width) {
    // Assign a LID to this operation
    getOrCreateOpLID(srcop);

    // Find the sort's associated lid in order to use it in the instruction
    size_t sid = sortToLIDMap.at(width);

    // Assuming that the operand has already been emitted
    // Find the LID associated to the operand
    size_t op0LID = getOrCreateOpLID(op0);

    // Build and return the slice instruction
    os << lid++ << wsStr << sliceStr << wsStr << sid << wsStr << op0LID << wsStr
       << (width - 1) << wsStr << lowbit << nlStr;
  }

  // Generates a constant declaration given a value, a width and a name
  void genUnaryOp(Operation *srcop, Operation *op0, StringRef inst,
                  size_t width) {
    // Register the source operation with the current line id
    getOrCreateOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Assuming that the operand has already been emitted
    // Find the LID associated to the operand
    size_t op0LID = getOrCreateOpLID(op0);

    os << lid++ << wsStr << inst << wsStr << sid << wsStr << op0LID << nlStr;
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
    size_t assertLID = getOrCreateOpLID(assertop);

    // Build and return the btor2 string
    os << lid++ << wsStr << badStr << wsStr << assertLID << nlStr;
  }

  // Generate a btor2 constraint given an expression from an assumption
  // operation
  void genConstraint(Value expr) {
    // Start by finding the expression lid
    size_t exprLID = getOrCreateOpLID(expr);

    // Build and return the btor2 string
    os << lid++ << wsStr << constraintStr << wsStr << exprLID << nlStr;
  }

  // Generate a btor2 constraint given an expression from an assumption
  // operation
  void genConstraint(size_t exprLID) {
    // Build and return the btor2 string
    os << lid++ << wsStr << constraintStr << wsStr << exprLID << nlStr;
  }

  // Generate an ite instruction (if then else) given a predicate, two values
  // and a res width
  void genIte(Operation *srcop, Value cond, Value t, Value f, int64_t width) {
    // Register the source operation with the current line id
    getOrCreateOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Retrieve the operand lids, assuming they were emitted
    size_t condLID = getOrCreateOpLID(cond);
    size_t tLID = getOrCreateOpLID(t);
    size_t fLID = getOrCreateOpLID(f);

    // Build and return the ite instruction
    os << lid++ << wsStr << iteStr << wsStr << sid << wsStr << condLID << wsStr
       << tLID << wsStr << fLID << nlStr;
  }

  // Generate an ite instruction (if then else) given a predicate, two values
  // and a res width
  void genIte(Operation *srcop, size_t condLID, size_t tLID, size_t fLID,
              int64_t width) {
    // Register the source operation with the current line id
    getOrCreateOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Build and return the ite instruction
    os << lid++ << wsStr << iteStr << wsStr << sid << wsStr << condLID << wsStr
       << tLID << wsStr << fLID << nlStr;
  }

  // Generate a logical implication given a lhs and a rhs
  void genImplies(Operation *srcop, Value lhs, Value rhs) {
    // Register the source operation with the current line id
    getOrCreateOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(1);

    // Retrieve LIDs for the lhs and rhs
    size_t lhsLID = getOrCreateOpLID(lhs);
    size_t rhsLID = getOrCreateOpLID(rhs);

    // Build and return the implies operation
    os << lid++ << wsStr << impliesStr << wsStr << sid << wsStr << lhsLID
       << wsStr << rhsLID << nlStr;
  }

  // Generate a logical implication given a lhs and a rhs
  void genImplies(Operation *srcop, size_t lhsLID, size_t rhsLID) {
    // Register the source operation with the current line id
    getOrCreateOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(1);

    // Build and return the implies operation
    os << lid++ << wsStr << impliesStr << wsStr << sid << wsStr << lhsLID
       << wsStr << rhsLID << nlStr;
  }

  // Generates a state instruction given a width and a name
  void genState(Operation *srcop, int64_t width, StringRef name) {
    // Register the source operation with the current line id
    getOrCreateOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Build and return the state instruction
    os << lid++ << wsStr << stateStr << wsStr << sid << wsStr << name << nlStr;
  }

  // Generates a next instruction, given a width, a state LID, and a next value
  // LID
  void genNext(Value next, Operation *reg, int64_t width) {
    // Retrieve the lid associated with the sort (sid)
    size_t sid = sortToLIDMap.at(width);

    // Retrieve the LIDs associated to reg and next
    size_t regLID = getOrCreateOpLID(reg);
    size_t nextLID = getOrCreateOpLID(next);

    // Build and return the next instruction
    os << lid++ << wsStr << nextStr << wsStr << sid << wsStr << regLID << wsStr
       << nextLID << nlStr;
  }

  // Verifies that the sort required for the given operation's btor2 emission
  // has been generated
  int64_t requireSort(mlir::Type type) {
    // Start by figuring out what sort needs to be generated
    int64_t width = hw::getBitWidth(type);
    assert(width != noWidth);

    // Generate the sort regardles of resulting width (nothing will be added if
    // the sort already exists)
    genSort(bitvecStr, width);
    return width;
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

      // Record the defining operation's line ID (the module itself in the case
      // of ports)
      inputLIDs[port.argNum] = lid;

      // We assume that the explicit name is always %reset for reset ports
      if (iName == resetStr)
        resetLID = lid;

      genInput(w, iName);
    }
  }

  // Outputs don't actually mean much in btor, only assertions matter
  // Additionally, btormc doesn't support outputs, so we're just going to
  // ignore them
  void visitTypeOp(hw::OutputOp op) {}

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
  void visitTypeOp(hw::WireOp op) {
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
  void visitBinOp(Operation *op, StringRef inst) {
    TypeSwitch<Operation *, void>(op)
        .template Case<
            // All supported binary ops
            comb::AddOp, comb::SubOp, comb::MulOp, comb::DivUOp, comb::DivSOp,
            comb::ModSOp, comb::ShlOp, comb::ShrUOp, comb::ShrSOp, comb::AndOp,
            comb::OrOp, comb::XorOp, comb::ConcatOp>([&](auto expr) {
          int64_t w = requireSort(expr.getType());

          // Start by extracting the operands
          Value op1 = expr->getOperand(0);
          Value op2 = expr->getOperand(1);

          // Generate the line
          genBinOp(inst, op, op1, op2, w);
        })
        // Ignore anything else
        .Default([&](auto expr) {});
  }

  // Visitors for the binary ops
  void visitComb(comb::AddOp op) { visitBinOp(op, addStr); }
  void visitComb(comb::SubOp op) { visitBinOp(op, subStr); }
  void visitComb(comb::MulOp op) { visitBinOp(op, mulStr); }
  void visitComb(comb::DivSOp op) { visitBinOp(op, sdivStr); }
  void visitComb(comb::DivUOp op) { visitBinOp(op, udivStr); }
  void visitComb(comb::ModSOp op) { visitBinOp(op, smodStr); }
  void visitComb(comb::ShlOp op) { visitBinOp(op, sllStr); }
  void visitComb(comb::ShrUOp op) { visitBinOp(op, srlStr); }
  void visitComb(comb::ShrSOp op) { visitBinOp(op, sraStr); }
  void visitComb(comb::AndOp op) { visitBinOp(op, andStr); }
  void visitComb(comb::OrOp op) { visitBinOp(op, orStr); }
  void visitComb(comb::XorOp op) { visitBinOp(op, xorStr); }
  void visitComb(comb::ConcatOp op) { visitBinOp(op, concatStr); }

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
    if (pred == hwNeqStr)
      pred = neqStr;

    // Width of result is always 1 for comparison
    genSort(bitvecStr, 1);

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
    genSort(bitvecStr, 1);

    // Check for an overaching enable
    // In our case the sv.if operation will probably only be used when
    // conditioning an sv.assert on an enable signal. This means that
    // its condition is probably used to imply our assertion
    if (auto ifop = dyn_cast<sv::IfOp>(((Operation *)op)->getParentOp())) {
      Value en = ifop.getOperand();

      // Generate the implication
      genImplies(ifop, en, expr);

      // Generate the implies inversion
      genUnaryOp(op, ifop, notStr, 1);
    } else {
      // Generate the expression inversion
      genUnaryOp(op, expr, notStr, 1);
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

  void visitInvalidSV(Operation *op) {
    // The only op left is registers
    auto reg = dyn_cast<seq::FirRegOp>(op);
    if (!reg)
      return;

    visit(reg);
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

    // Visit all of the operations in our module
    module.walk([&](Operation *op) { dispatchTypeOpVisitor(op); });

    // Iterate through the registers and generate the `next` instructions
    for (size_t i = 0; i < regOps.size(); ++i) {
      // Check the register type (done to support non-firrtl registers as well)
      auto reg = dyn_cast<seq::FirRegOp>(regOps[i]);
      if (!reg)
        continue;

      // Generate the reset condition (for sync & async resets)
      // We assume for now that the reset value is always 0
      size_t width = hw::getBitWidth(reg.getType());
      genSort(bitvecStr, width);
      genZero(width);

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
        if (inputLIDs.contains(argIdx)) {
          nextLID = inputLIDs[argIdx];
        }
      } else {
        nextLID = getOrCreateOpLID(next);
      }

      // Sanity check
      assert(nextLID != noLID);

      // Generate the ite for the register update reset condition
      // i.e. reg <= reset ? 0 : next
      genIte(next.getDefiningOp(), resetLID, constToLIDMap.at({0, width}),
             nextLID, width);

      // Finally generate the next statement
      genNext(next, reg, width);
    }
  });

  // Clear data structures to allow for pass reuse
  sortToLIDMap.clear();
  constToLIDMap.clear();
  opLIDMap.clear();
  opAliasMap.clear();
  inputLIDs.clear();
  regOps.clear();
}

// Basic constructor for the pass
std::unique_ptr<mlir::Pass> circt::createConvertHWToBTOR2Pass() {
  return std::make_unique<ConvertHWToBTOR2Pass>(llvm::outs());
}
