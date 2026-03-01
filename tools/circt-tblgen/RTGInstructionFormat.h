//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the instruction format parser and code generators for
// RTG instructions. It provides AST nodes for representing instruction format
// specifications and generates assembly printing and binary encoding methods.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/LLVM.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/Support/Allocator.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

namespace circt {
namespace tblgen {

//===----------------------------------------------------------------------===//
// Operand Type Classification
//===----------------------------------------------------------------------===//

/// Classification of operand types in RTG instructions.
/// Used to distinguish between different kinds of operands during code
/// generation and validation.
enum class OperandType { Register, Immediate, Label };

//===----------------------------------------------------------------------===//
// Format AST
//===----------------------------------------------------------------------===//

/// Base class for all format AST nodes.
/// Represents a single element in an instruction format specification,
/// such as operands, literals, mnemonics, or signedness specifiers.
class FormatNode {
public:
  enum class Kind {
    /// An operand reference (e.g., $rs, $imm[11:0])
    Operand,
    /// A binary literal value (e.g., 0b1010)
    BinaryLiteral,
    /// A string literal (e.g., `add`)
    StringLiteral,
    /// The mnemonic keyword placeholder
    Mnemonic,
    /// A signedness wrapper (signed()/unsigned())
    SignednessSpecifier
  };

  FormatNode(Kind kind, llvm::SMLoc loc, StringRef desc)
      : kind(kind), loc(loc), desc(desc) {}
  virtual ~FormatNode() = default;

  Kind getKind() const { return kind; }

  /// Get the source location of this node in the TableGen file.
  llvm::SMLoc getLoc() const { return loc; }

  /// Get a human-readable description of this node type.
  StringRef getDesc() const { return desc; }

private:
  Kind kind;
  llvm::SMLoc loc;
  StringRef desc;
};

/// Represents an operand reference in an instruction format.
/// Supports optional bit slicing for extracting specific bit ranges from
/// operands (e.g., $imm[11:0] extracts bits 11 through 0 (both inclusive) of
/// the immediate).
class OperandNode : public FormatNode {
public:
  /// Construct an operand node.
  /// \param loc Source location in the TableGen file
  /// \param name Name of the operand (without the leading $)
  /// \param highBit High bit of the slice, or -1 if no slicing
  /// \param lowBit Low bit of the slice, or -1 if no slicing
  OperandNode(llvm::SMLoc loc, StringRef name, int highBit = -1,
              int lowBit = -1)
      : FormatNode(Kind::Operand, loc, "operand"), name(name), highBit(highBit),
        lowBit(lowBit) {}

  /// Get the name of the operand (without the leading $).
  StringRef getName() const { return name; }

  /// Check if this operand has a bit slice specification.
  bool hasBitSlice() const { return highBit >= 0 && lowBit >= 0; }

  /// Get the high bit of the slice (inclusive).
  int getHighBit() const { return highBit; }

  /// Get the low bit of the slice (inclusive).
  int getLowBit() const { return lowBit; }

  /// Get the width of the bit slice in bits.
  int getBitWidth() const { return highBit - lowBit + 1; }

  static bool classof(const FormatNode *elem) {
    return elem->getKind() == Kind::Operand;
  }

  /// The set of possible operand types for this operand.
  /// Only valid after `resolveOperands()` has been called on the context.
  /// An operand may have multiple possible types, e.g., if `AnyTypeOf` is used.
  DenseSet<OperandType> kinds;
  /// The parent node of this operand node. 'nullptr' if this is a top-level
  /// operand.
  FormatNode *parent = nullptr;
  /// The binary encoding width for register operands.
  /// Only valid after `resolveOperands()` has been called and if this is a
  /// register operand. Set to -1 if not a register or not yet resolved.
  int registerBinaryEncodingWidth = -1;

private:
  SmallString<32> name;
  int highBit;
  int lowBit;
};

/// Represents a binary literal value in an instruction format.
/// Used in binary format specifications to represent fixed bit patterns
/// (e.g., 0b1010, 0b0010011). These are typically opcode bits or other
/// constant fields in the instruction encoding.
class BinaryLiteralNode : public FormatNode {
public:
  /// Construct a binary literal node.
  /// \param loc Source location in the TableGen file
  /// \param value The binary value as an APInt
  BinaryLiteralNode(llvm::SMLoc loc, const APInt &value)
      : FormatNode(Kind::BinaryLiteral, loc, "binary literal"), value(value) {}

  /// Get the binary value.
  const APInt &getValue() const { return value; }

  /// Get the width of the binary literal in bits.
  unsigned getWidth() const { return value.getBitWidth(); }

  static bool classof(const FormatNode *elem) {
    return elem->getKind() == Kind::BinaryLiteral;
  }

private:
  APInt value;
};

/// Represents a string literal in an assembly format.
/// Used to specify fixed text that appears in the assembly representation
/// of an instruction (e.g., `add`, `,`, `(`). String literals are enclosed
/// in backticks in the format specification.
class StringLiteralNode : public FormatNode {
public:
  /// Construct a string literal node.
  /// \param loc Source location in the TableGen file
  /// \param str The string literal content (without backticks)
  StringLiteralNode(llvm::SMLoc loc, StringRef str)
      : FormatNode(Kind::StringLiteral, loc, "string literal"), str(str) {}

  /// Get the string literal content.
  StringRef getLiteral() const { return str; }

  static bool classof(const FormatNode *elem) {
    return elem->getKind() == Kind::StringLiteral;
  }

private:
  SmallString<32> str;
};

/// Represents the mnemonic placeholder in an assembly format.
/// The 'mnemonic' keyword in a format specification is replaced with the
/// actual instruction mnemonic (e.g., "add", "sub", "lw") during code
/// generation.
struct MnemonicNode : public FormatNode {
  /// Construct a mnemonic node.
  /// \param loc Source location in the TableGen file
  MnemonicNode(llvm::SMLoc loc)
      : FormatNode(Kind::Mnemonic, loc, "'mnemonic' keyword") {}

  static bool classof(const FormatNode *elem) {
    return elem->getKind() == Kind::Mnemonic;
  }
};

/// Represents a signedness specifier for an operand in an assembly format.
/// Wraps an operand to indicate whether it should be printed as signed or
/// unsigned (e.g., signed($imm) or unsigned($imm)).
class SignednessNode : public FormatNode {
public:
  /// Construct a signedness specifier node.
  /// \param loc Source location in the TableGen file
  /// \param isSigned True for signed(), false for unsigned()
  /// \param operand The operand being wrapped
  SignednessNode(llvm::SMLoc loc, bool isSigned, OperandNode *operand)
      : FormatNode(Kind::SignednessSpecifier, loc, "signed()/unsigned()"),
        isSgnd(isSigned), operand(operand) {}

  /// Check if this is a signed() specifier.
  bool isSigned() const { return isSgnd; }

  /// Check if this is an unsigned() specifier.
  bool isUnsigned() const { return !isSgnd; }

  /// Get the operand being wrapped.
  OperandNode *getOperand() const { return operand; }

  static bool classof(const FormatNode *elem) {
    return elem->getKind() == Kind::SignednessSpecifier;
  }

private:
  bool isSgnd;
  OperandNode *operand;
};

/// Context for managing the format AST for a single instruction operation.
/// Provides memory allocation for AST nodes and maintains the list of root
/// nodes in the format specification. Each ASTContext is associated with
/// a specific MLIR operation being processed.
class ASTContext {
public:
  /// Construct an AST context for the given operation.
  /// \param op The MLIR TableGen operator being processed
  explicit ASTContext(mlir::tblgen::Operator &op) : op(op) {}

  /// Allocate and construct a format node using the internal allocator.
  /// \tparam T The type of node to create (must derive from FormatNode)
  /// \param args Arguments to forward to the node's constructor
  /// \return Pointer to the newly created node
  template <typename T, typename... Args>
  T *create(Args &&...args) {
    return new (allocator.Allocate<T>()) T(std::forward<Args>(args)...);
  }

  /// Add a node to the list of root nodes in the format.
  void addNode(FormatNode *node) { rootNodes.push_back(node); }

  /// Get an iterator range over all root nodes in the format.
  auto nodes() const {
    return llvm::make_range(rootNodes.begin(), rootNodes.end());
  }

  /// Get the MLIR operator associated with this context.
  const mlir::tblgen::Operator &getOp() { return op; }

private:
  llvm::BumpPtrAllocator allocator;
  SmallVector<FormatNode *> rootNodes;
  mlir::tblgen::Operator &op;
};

/// Base class for instruction format code generators.
/// Provides a common interface for generating different types of instruction
/// format methods (assembly printing, binary encoding, etc.) from a format AST.
struct FormatGen {
  virtual ~FormatGen() = default;

  /// Generate an instruction method for the given operation.
  /// \param ctx The AST context containing the format specification
  /// \param opClassName The name of the operation class this method is being
  ///                    generated for.
  virtual void genInstructionMethod(ASTContext &ctx, StringRef opClassName) = 0;
};

/// Code generator for assembly format printing methods.
/// Generates C++ code that implements the ISA assembly format printer for
/// an instruction operation based on its format AST.
struct AssemblyFormatGen : public FormatGen {
  /// Construct an assembly format generator.
  /// \param os Output stream to write generated code to
  explicit AssemblyFormatGen(raw_ostream &os) : os(os) {}

  void genInstructionMethod(ASTContext &ctx, StringRef opClassName) override;

private:
  void gen(StringLiteralNode *node);
  void gen(MnemonicNode *node);
  void gen(OperandNode *node);
  void gen(SignednessNode *node);

  raw_ostream &os;
};

/// Code generator for binary format encoding methods.
/// Generates C++ code that implements the binary encoding for an instruction
/// operation based on its binary format AST. Handles bit packing of operands
/// and literal values.
struct BinaryFormatGen : public FormatGen {
  /// Construct a binary format generator.
  /// \param os Output stream to write generated code to
  explicit BinaryFormatGen(raw_ostream &os) : os(os) {}

  void genInstructionMethod(ASTContext &ctx, StringRef opClassName) override;

private:
  void gen(BinaryLiteralNode *node);
  // genDecl must have been called before gen() is called.
  void gen(OperandNode *node);
  // Generate variable declarations for an operand node.
  void genDecl(OperandNode *node);

  DenseSet<StringRef> decls;
  raw_ostream &os;
};

/// Parser for instruction format strings.
/// Parses both binary and assembly format specifications from TableGen
/// definitions and builds an AST representation.
struct FormatParser {
  /// Construct a format parser.
  /// \param ctx The AST context to populate with parsed nodes
  FormatParser(ASTContext &ctx) : ctx(ctx) {}

  /// Parse a format string and append the resulting nodes to the context.
  /// \param loc Source location of the format string in the TableGen file
  /// \param format The format string to parse
  void parseAndAppendToContext(llvm::SMLoc loc, StringRef format);

  /// Parses the "mnemonic" keyword in the format.
  /// \return A MnemonicNode if found, nullptr otherwise
  FormatNode *parseOptionalMnemonic();

  /// Parses operand references like "$rs" or "$imm[11:0]" with optional bit
  /// slicing. Does not resolve the MLIR type of the operand.
  /// \return An OperandNode if found, nullptr otherwise
  FormatNode *parseOptionalOperand();

  /// Parses string literals enclosed in backticks (e.g., "`add`", "`,`").
  /// \return A StringLiteralNode if found, nullptr otherwise
  FormatNode *parseOptionalStringLiteral();

  /// Parses binary literals in the form "0b1010" or "0b0010011".
  /// \return A BinaryLiteralNode if found, nullptr otherwise
  FormatNode *parseOptionalBinaryLiteral();

  /// Parses "signed($operand)" or "unsigned($operand)" wrappers.
  /// \return A SignednessNode if found, nullptr otherwise
  FormatNode *parseOptionalSignednessSpecifier();

  /// Parses bit slices like "[11:0]" or "[5]" (single bit).
  /// \param highBit Output parameter for the high bit (inclusive)
  /// \param lowBit Output parameter for the low bit (inclusive)
  /// \return True if a slice was parsed, false otherwise.
  ///         The output parameters are not modified if no slice was found.
  bool parseOptionalSlice(int64_t &highBit, int64_t &lowBit);

  /// Skip whitespace at the current parsing position.
  /// Advances the position to the next non-whitespace character.
  void skipWhitespace();

  /// Consume an expected string at the current position.
  /// Emits an error if the string does not match.
  /// \param str The string to consume
  /// \param skipWhitespaceFirst If true, skip whitespace before matching
  void consume(StringRef str, bool skipWhitespaceFirst = true);

  /// Try to consume a string at the current position.
  /// \param str The string to try to consume
  /// \param skipWhitespaceFirst If true, skip whitespace before matching
  /// \return True if the string was consumed, false otherwise
  bool tryConsume(StringRef str, bool skipWhitespaceFirst = true);

  /// Parse an integer literal at the current position.
  /// \return The parsed integer value
  int64_t parseIntLiteral();

  /// Parses identifiers like "$rs" or "$imm".
  /// \return The identifier without the leading "$"
  StringRef parseIdentifier();

  /// Get the current source location in the input string.
  /// \return The current source location for error reporting
  llvm::SMLoc getLoc();

private:
  ASTContext &ctx;
  SmallString<128> input;
  size_t pos = 0;
  llvm::SMLoc loc;
};

/// Verify that a format AST is valid for assembly format generation.
/// Checks that:
/// - The format does not contain binary literals (only valid in binary
///   formats)
/// - The format does not contain operand bit slicing (only valid in binary
///   formats)
/// - Immediate operands are wrapped in signed()/unsigned() specifiers
/// - Label and register operands are not wrapped in signedness specifiers
///
/// Must be called after resolveOperands() has been called on the context.
/// \param ctx The AST context to verify
void verifyAssemblyFormat(ASTContext &ctx);

/// Verify that a format AST is valid for binary format generation.
/// Checks that:
/// - The format does not contain mnemonics (only valid in assembly formats)
/// - The format does not contain string literals (only valid in assembly
///   formats)
/// - The format does not contain signedness specifiers (only valid in assembly
///   formats)
///
/// \param ctx The AST context to verify
void verifyBinaryFormat(ASTContext &ctx);

/// Generate print and encoding methods for an instruction format operation.
/// This is the main entry point for generating code from an instruction
/// format specification. It processes both assembly and binary formats if
/// present in the operation definition.
///
/// \param opDef The TableGen record defining the instruction operation
/// \param os Output stream to write generated code to
void genInstructionPrintMethods(const llvm::Record *opDef, raw_ostream &os);

} // namespace tblgen
} // namespace circt
