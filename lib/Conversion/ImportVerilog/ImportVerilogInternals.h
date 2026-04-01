//===- ImportVerilogInternals.h - Internal implementation details ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H
#define CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H

#include "circt/Conversion/ImportVerilog.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "slang/ast/ASTVisitor.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <queue>

#define DEBUG_TYPE "import-verilog"

namespace circt {
namespace ImportVerilog {

using moore::Domain;

/// Port lowering information.
struct PortLowering {
  const slang::ast::PortSymbol &ast;
  Location loc;
  BlockArgument arg;
};

/// Lowering information for a single signal flattened from an interface port.
struct FlattenedIfacePort {
  StringAttr name;
  hw::ModulePort::Direction direction;
  mlir::Type type;
  Location loc;
  BlockArgument arg;
  /// the origin interface port symbol this was flattened from.
  const slang::ast::InterfacePortSymbol *origin;
  /// the interface body member (VariableSymbol , NetSymbol)
  const slang::ast::Symbol *bodySym;
  /// The connected interface instance backing this port (if any). This enables
  /// materializing virtual interface handles from interface ports.
  const slang::ast::InstanceSymbol *ifaceInstance = nullptr;
};

/// Lowering information for an expanded interface instance. Maps each interface
/// body member to its expanded SSA value (moore.variable or moore.net).
struct InterfaceLowering {
  DenseMap<const slang::ast::Symbol *, Value> expandedMembers;
  DenseMap<StringAttr, Value> expandedMembersByName;
};

/// Cached lowering information for representing SystemVerilog `virtual
/// interface` handles as Moore types (a struct of references to interface
/// members).
struct VirtualInterfaceLowering {
  moore::UnpackedStructType type;
  SmallVector<StringAttr, 8> fieldNames;
};

/// A mapping entry for resolving Slang virtual interface member accesses.
///
/// Slang may resolve `vif.member` expressions (where `vif` has a
/// `VirtualInterfaceType`) directly to a `NamedValueExpression` for `member`.
/// This table records which virtual interface base symbol that member access is
/// rooted in, so ImportVerilog can materialize the appropriate Moore IR.
struct VirtualInterfaceMemberAccess {
  const slang::ast::ValueSymbol *base = nullptr;
  /// The name of the field in the lowered virtual interface handle struct that
  /// should be accessed for this member.
  StringAttr fieldName;
};

/// Module lowering information.
struct ModuleLowering {
  moore::SVModuleOp op;
  SmallVector<PortLowering> ports;
  SmallVector<FlattenedIfacePort> ifacePorts;
  DenseMap<const slang::syntax::SyntaxNode *, const slang::ast::PortSymbol *>
      portsBySyntaxNode;
};

/// Function lowering information. The `op` field holds either a `func::FuncOp`
/// (for SystemVerilog functions) or a `moore::CoroutineOp` (for tasks),
/// accessed through the `FunctionOpInterface`.
struct FunctionLowering {
  mlir::FunctionOpInterface op;
  llvm::SmallVector<Value, 4> captures;
  llvm::DenseMap<Value, unsigned> captureIndex;
  bool capturesFinalized = false;
  bool isConverting = false;

  /// Whether this is a coroutine (task) or a regular function.
  bool isCoroutine() { return isa<moore::CoroutineOp>(op.getOperation()); }
};

// Class lowering information.
struct ClassLowering {
  circt::moore::ClassDeclOp op;
  bool methodsFinalized = false;
};

/// Information about a loops continuation and exit blocks relevant while
/// lowering the loop's body statements.
struct LoopFrame {
  /// The block to jump to from a `continue` statement.
  Block *continueBlock;
  /// The block to jump to from a `break` statement.
  Block *breakBlock;
};

/// Hierarchical path information.
/// The "hierName" means a different hierarchical name at different module
/// levels.
/// The "idx" means where the current hierarchical name is on the portlists.
/// The "direction" means hierarchical names whether downward(In) or
/// upward(Out).
struct HierPathInfo {
  mlir::StringAttr hierName;
  std::optional<unsigned int> idx;
  slang::ast::ArgumentDirection direction;
  const slang::ast::ValueSymbol *valueSym;
};

/// A helper class to facilitate the conversion from a Slang AST to MLIR
/// operations. Keeps track of the destination MLIR module, builders, and
/// various worklists and utilities needed for conversion.
struct Context {
  Context(const ImportVerilogOptions &options,
          slang::ast::Compilation &compilation, mlir::ModuleOp intoModuleOp,
          const slang::SourceManager &sourceManager)
      : options(options), compilation(compilation), intoModuleOp(intoModuleOp),
        sourceManager(sourceManager),
        builder(OpBuilder::atBlockEnd(intoModuleOp.getBody())),
        symbolTable(intoModuleOp) {}
  Context(const Context &) = delete;

  /// Return the MLIR context.
  MLIRContext *getContext() { return intoModuleOp.getContext(); }

  /// Convert a slang `SourceLocation` into an MLIR `Location`.
  Location convertLocation(slang::SourceLocation loc);
  /// Convert a slang `SourceRange` into an MLIR `Location`.
  Location convertLocation(slang::SourceRange range);

  /// Convert a slang type into an MLIR type. Returns null on failure. Uses the
  /// provided location for error reporting, or tries to guess one from the
  /// given type. Types tend to have unreliable location information, so it's
  /// generally a good idea to pass in a location.
  Type convertType(const slang::ast::Type &type, LocationAttr loc = {});
  Type convertType(const slang::ast::DeclaredType &type);

  /// Convert hierarchy and structure AST nodes to MLIR ops.
  LogicalResult convertCompilation();
  ModuleLowering *
  convertModuleHeader(const slang::ast::InstanceBodySymbol *module);
  LogicalResult convertModuleBody(const slang::ast::InstanceBodySymbol *module);
  LogicalResult convertPackage(const slang::ast::PackageSymbol &package);
  FunctionLowering *
  declareFunction(const slang::ast::SubroutineSymbol &subroutine);
  LogicalResult convertFunction(const slang::ast::SubroutineSymbol &subroutine);
  LogicalResult finalizeFunctionBodyCaptures(FunctionLowering &lowering);
  ClassLowering *declareClass(const slang::ast::ClassType &cls);
  LogicalResult buildClassProperties(const slang::ast::ClassType &classdecl);
  LogicalResult materializeClassMethods(const slang::ast::ClassType &classdecl);
  LogicalResult convertGlobalVariable(const slang::ast::VariableSymbol &var);

  /// Convert a Slang virtual interface type into the Moore type used to
  /// represent virtual interface handles. Populates internal caches so that
  /// interface instance references can be materialized consistently.
  FailureOr<moore::UnpackedStructType>
  convertVirtualInterfaceType(const slang::ast::VirtualInterfaceType &type,
                              Location loc);

  /// Materialize a Moore value representing a concrete interface instance as a
  /// virtual interface handle. This only succeeds for the Slang
  /// `VirtualInterfaceType` wrappers that refer to a real interface instance
  /// (`isRealIface`).
  FailureOr<Value>
  materializeVirtualInterfaceValue(const slang::ast::VirtualInterfaceType &type,
                                   Location loc);

  /// Register the interface members of a virtual interface base symbol for use
  /// in later expression conversion.
  LogicalResult
  registerVirtualInterfaceMembers(const slang::ast::ValueSymbol &base,
                                  const slang::ast::VirtualInterfaceType &type,
                                  Location loc);

  /// Checks whether one class (actualTy) is derived from another class
  /// (baseTy). True if it's a subclass, false otherwise.
  bool isClassDerivedFrom(const moore::ClassHandleType &actualTy,
                          const moore::ClassHandleType &baseTy);

  /// Tries to find the closest base class of actualTy that carries a property
  /// with name fieldName. The location is used for error reporting.
  moore::ClassHandleType
  getAncestorClassWithProperty(const moore::ClassHandleType &actualTy,
                               StringRef fieldName, Location loc);

  Value getImplicitThisRef() const {
    return currentThisRef; // block arg added in declareFunction
  }

  Value getIndexedQueue() const { return currentQueue; }

  // Convert a statement AST node to MLIR ops.
  LogicalResult convertStatement(const slang::ast::Statement &stmt);

  // Convert an expression AST node to MLIR ops.
  Value convertRvalueExpression(const slang::ast::Expression &expr,
                                Type requiredType = {});
  Value convertLvalueExpression(const slang::ast::Expression &expr);

  // Convert an assertion expression AST node to MLIR ops.
  Value convertAssertionExpression(const slang::ast::AssertionExpr &expr,
                                   Location loc);

  // Convert an assertion expression AST node to MLIR ops.
  Value convertAssertionCallExpression(
      const slang::ast::CallExpression &expr,
      const slang::ast::CallExpression::SystemCallInfo &info, Location loc);

  // Traverse the whole AST to collect hierarchical names.
  void traverseInstanceBody(const slang::ast::Symbol &symbol);

  // Convert timing controls into a corresponding set of ops that delay
  // execution of the current block. Produces an error if the implicit event
  // control `@*` or `@(*)` is used.
  LogicalResult convertTimingControl(const slang::ast::TimingControl &ctrl);
  // Convert timing controls into a corresponding set of ops that delay
  // execution of the current block. Then converts the given statement, taking
  // note of the rvalues it reads and adding them to a wait op in case an
  // implicit event control `@*` or `@(*)` is used.
  LogicalResult convertTimingControl(const slang::ast::TimingControl &ctrl,
                                     const slang::ast::Statement &stmt);

  /// Helper function to convert a value to a MLIR I1 value.
  Value convertToI1(Value value);

  // Convert a slang timing control for LTL
  Value convertLTLTimingControl(const slang::ast::TimingControl &ctrl,
                                const Value &seqOrPro);

  /// Helper function to convert a value to its "truthy" boolean value.
  Value convertToBool(Value value);

  /// Helper function to convert a value to its "truthy" boolean value and
  /// convert it to the given domain.
  Value convertToBool(Value value, Domain domain);

  /// Helper function to convert a value to its simple bit vector
  /// representation, if it has one. Otherwise returns null. Also returns null
  /// if the given value is null.
  Value convertToSimpleBitVector(Value value);

  /// Helper function to insert the necessary operations to cast a value from
  /// one type to another.
  Value materializeConversion(Type type, Value value, bool isSigned,
                              Location loc);

  /// Helper function to materialize an `SVInt` as an SSA value.
  Value materializeSVInt(const slang::SVInt &svint,
                         const slang::ast::Type &type, Location loc);

  /// Helper function to materialize a real value as an SSA value.
  Value materializeSVReal(const slang::ConstantValue &svreal,
                          const slang::ast::Type &type, Location loc);

  /// Helper function to materialize a string as an SSA value.
  Value materializeString(const slang::ConstantValue &string,
                          const slang::ast::Type &astType, Location loc);

  /// Helper function to materialize an unpacked array of `SVInt`s as an SSA
  /// value.
  Value materializeFixedSizeUnpackedArrayType(
      const slang::ConstantValue &constant,
      const slang::ast::FixedSizeUnpackedArrayType &astType, Location loc);

  /// Helper function to materialize a `ConstantValue` as an SSA value. Returns
  /// null if the constant cannot be materialized.
  Value materializeConstant(const slang::ConstantValue &constant,
                            const slang::ast::Type &type, Location loc);

  /// Convert a list of string literal arguments with formatting specifiers and
  /// arguments to be interpolated into a `!moore.format_string` value. Returns
  /// failure if an error occurs. Returns a null value if the formatted string
  /// is trivially empty. Otherwise returns the formatted string.
  FailureOr<Value> convertFormatString(
      std::span<const slang::ast::Expression *const> arguments, Location loc,
      moore::IntFormat defaultFormat = moore::IntFormat::Decimal,
      bool appendNewline = false);

  /// Convert system function calls. Returns a null `Value` on failure after
  /// emitting an error.
  Value convertSystemCall(const slang::ast::SystemSubroutine &subroutine,
                          Location loc,
                          std::span<const slang::ast::Expression *const> args);

  /// Convert system function calls within properties and assertion with a
  /// single argument.
  FailureOr<Value> convertAssertionSystemCallArity1(
      const slang::ast::SystemSubroutine &subroutine, Location loc, Value value,
      Type originalType);

  /// Evaluate the constant value of an expression.
  slang::ConstantValue evaluateConstant(const slang::ast::Expression &expr);

  const ImportVerilogOptions &options;
  slang::ast::Compilation &compilation;
  mlir::ModuleOp intoModuleOp;
  const slang::SourceManager &sourceManager;

  /// The builder used to create IR operations.
  OpBuilder builder;
  /// A symbol table of the MLIR module we are emitting into.
  SymbolTable symbolTable;

  /// The top-level operations ordered by their Slang source location. This is
  /// used to produce IR that follows the source file order.
  std::map<slang::SourceLocation, Operation *> orderedRootOps;

  /// How we have lowered modules to MLIR.
  DenseMap<const slang::ast::InstanceBodySymbol *,
           std::unique_ptr<ModuleLowering>>
      modules;

  /// Expanded interface instances, keyed by the InstanceSymbol pointer.
  /// Each entry maps body members to their expanded SSA values. Scoped
  /// per-module so entries are cleaned up when a module's conversion ends.
  using InterfaceInstances =
      llvm::ScopedHashTable<const slang::ast::InstanceSymbol *,
                            InterfaceLowering *>;
  using InterfaceInstanceScope = InterfaceInstances::ScopeTy;
  InterfaceInstances interfaceInstances;
  /// Owning storage for InterfaceLowering objects
  /// because ScopedHashTable stores values by copy.
  SmallVector<std::unique_ptr<InterfaceLowering>> interfaceInstanceStorage;

  /// Cached virtual interface layouts (type + field order).
  DenseMap<const slang::ast::InstanceBodySymbol *, VirtualInterfaceLowering>
      virtualIfaceLowerings;
  DenseMap<const slang::ast::ModportSymbol *, VirtualInterfaceLowering>
      virtualIfaceModportLowerings;
  /// A list of modules for which the header has been created, but the body has
  /// not been converted yet.
  std::queue<const slang::ast::InstanceBodySymbol *> moduleWorklist;

  /// Functions that have already been converted.
  DenseMap<const slang::ast::SubroutineSymbol *,
           std::unique_ptr<FunctionLowering>>
      functions;

  /// Classes that have already been converted.
  DenseMap<const slang::ast::ClassType *, std::unique_ptr<ClassLowering>>
      classes;

  /// A table of defined values, such as variables, that may be referred to by
  /// name in expressions. The expressions use this table to lookup the MLIR
  /// value that was created for a given declaration in the Slang AST node.
  using ValueSymbols =
      llvm::ScopedHashTable<const slang::ast::ValueSymbol *, Value>;
  using ValueSymbolScope = ValueSymbols::ScopeTy;
  ValueSymbols valueSymbols;

  /// A table mapping symbols for interface members accessed through a virtual
  /// interface to the virtual interface base value symbol.
  using VirtualInterfaceMembers =
      llvm::ScopedHashTable<const slang::ast::ValueSymbol *,
                            VirtualInterfaceMemberAccess>;
  using VirtualInterfaceMemberScope = VirtualInterfaceMembers::ScopeTy;
  VirtualInterfaceMembers virtualIfaceMembers;

  /// A table of defined global variables that may be referred to by name in
  /// expressions.
  DenseMap<const slang::ast::ValueSymbol *, moore::GlobalVariableOp>
      globalVariables;
  /// A list of global variables that still need their initializers to be
  /// converted.
  SmallVector<const slang::ast::ValueSymbol *> globalVariableWorklist;

  /// Collect all hierarchical names used for the per module/instance.
  DenseMap<const slang::ast::InstanceBodySymbol *, SmallVector<HierPathInfo>>
      hierPaths;

  /// It's used to collect the repeat hierarchical names on the same path.
  /// Such as `Top.sub.a` and `sub.a`, they are equivalent. The variable "a"
  /// will be added to the port list. But we only record once. If we don't do
  /// that. We will view the strange IR, such as `module @Sub(out y, out y)`;
  DenseSet<StringAttr> sameHierPaths;

  /// A stack of assignment left-hand side values. Each assignment will push its
  /// lowered left-hand side onto this stack before lowering its right-hand
  /// side. This allows expressions to resolve the opaque
  /// `LValueReferenceExpression`s in the AST.
  SmallVector<Value> lvalueStack;

  /// A stack of loop continuation and exit blocks. Each loop will push the
  /// relevant info onto this stack, lower its loop body statements, and pop the
  /// info off the stack again. Continue and break statements encountered as
  /// part of the loop body statements will use this information to branch to
  /// the correct block.
  SmallVector<LoopFrame> loopStack;

  /// A listener called for every variable or net being read. This can be used
  /// to collect all variables read as part of an expression or statement, for
  /// example to populate the list of observed signals in an implicit event
  /// control `@*`.
  std::function<void(moore::ReadOp)> rvalueReadCallback;
  /// A listener called for every variable or net being assigned. This can be
  /// used to collect all variables assigned in a task scope.
  std::function<void(mlir::Operation *)> variableAssignCallback;

  /// Whether we are currently converting expressions inside a timing control,
  /// such as `@(posedge clk)`. This is used by the implicit event control
  /// callback to avoid adding reads from explicit event controls to the
  /// implicit sensitivity list.
  bool isInsideTimingControl = false;

  /// The time scale currently in effect.
  slang::TimeScale timeScale;

  /// Variable to track the value of the current function's implicit `this`
  /// reference
  Value currentThisRef = {};

  /// Variable that tracks the queue which we are currently converting the index
  /// expression for. This is necessary to implement the `$` operator, which
  /// returns the index of the last element of the queue.
  Value currentQueue = {};

  /// Ensure that the global variables for `$monitor` state exist. This creates
  /// the `__monitor_active_id` and `__monitor_enabled` globals on first call.
  void ensureMonitorGlobals();

  /// Process any pending `$monitor` calls and generate the monitoring
  /// procedures at module level.
  LogicalResult flushPendingMonitors();

  /// Global variable ops for `$monitor` state management. These are created on
  /// demand by `ensureMonitorGlobals()`.
  moore::GlobalVariableOp monitorActiveIdGlobal = nullptr;
  moore::GlobalVariableOp monitorEnabledGlobal = nullptr;

  /// The next monitor ID to allocate. ID 0 is reserved for "no monitor active".
  unsigned nextMonitorId = 1;

  /// Information about a pending `$monitor` call that needs to be converted
  /// after the current module's body has been processed.
  struct PendingMonitor {
    unsigned id;
    Location loc;
    const slang::ast::CallExpression *call;
  };

  /// Pending `$monitor` calls that need to be converted at module level.
  SmallVector<PendingMonitor> pendingMonitors;

private:
  /// Helper function to extract the commonalities in lowering of functions and
  /// methods
  FunctionLowering *
  declareCallableImpl(const slang::ast::SubroutineSymbol &subroutine,
                      mlir::StringRef qualifiedName,
                      llvm::SmallVectorImpl<Type> &extraParams);
};

} // namespace ImportVerilog
} // namespace circt
#endif // CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H
