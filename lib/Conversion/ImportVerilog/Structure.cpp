//===- Structure.cpp - Slang hierarchy conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"

using namespace circt;
using namespace ImportVerilog;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static void guessNamespacePrefix(const slang::ast::Symbol &symbol,
                                 SmallString<64> &prefix) {
  if (symbol.kind != slang::ast::SymbolKind::Package)
    return;
  guessNamespacePrefix(symbol.getParentScope()->asSymbol(), prefix);
  if (!symbol.name.empty()) {
    prefix += symbol.name;
    prefix += "::";
  }
}

//===----------------------------------------------------------------------===//
// Base Visitor
//===----------------------------------------------------------------------===//

namespace {
/// Base visitor which ignores AST nodes that are handled by Slang's name
/// resolution and type checking.
struct BaseVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  BaseVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  // Skip semicolons.
  LogicalResult visit(const slang::ast::EmptyMemberSymbol &) {
    return success();
  }

  // Skip members that are implicitly imported from some other scope for the
  // sake of name resolution, such as enum variant names.
  LogicalResult visit(const slang::ast::TransparentMemberSymbol &) {
    return success();
  }

  // Skip typedefs.
  LogicalResult visit(const slang::ast::TypeAliasType &) { return success(); }
  LogicalResult visit(const slang::ast::ForwardingTypedefSymbol &) {
    return success();
  }

  // Skip imports. The AST already has its names resolved.
  LogicalResult visit(const slang::ast::ExplicitImportSymbol &) {
    return success();
  }
  LogicalResult visit(const slang::ast::WildcardImportSymbol &) {
    return success();
  }

  // Skip type parameters. The Slang AST is already monomorphized.
  LogicalResult visit(const slang::ast::TypeParameterSymbol &) {
    return success();
  }

  // Skip elaboration system tasks. These are reported directly by Slang.
  LogicalResult visit(const slang::ast::ElabSystemTaskSymbol &) {
    return success();
  }

  // Handle parameters.
  LogicalResult visit(const slang::ast::ParameterSymbol &param) {
    visitParameter(param);
    return success();
  }

  LogicalResult visit(const slang::ast::SpecparamSymbol &param) {
    visitParameter(param);
    return success();
  }

  template <class Node>
  void visitParameter(const Node &param) {
    // If debug info is enabled, try to materialize the parameter's constant
    // value on a best-effort basis and create a `dbg.variable` to track the
    // value.
    if (!context.options.debugInfo)
      return;
    auto value =
        context.materializeConstant(param.getValue(), param.getType(), loc);
    if (!value)
      return;
    if (builder.getInsertionBlock()->getParentOp() == context.intoModuleOp)
      context.orderedRootOps.insert({param.location, value.getDefiningOp()});

    // Prefix the parameter name with the surrounding namespace to create
    // somewhat sane names in the IR.
    SmallString<64> paramName;
    guessNamespacePrefix(param.getParentScope()->asSymbol(), paramName);
    paramName += param.name;

    builder.create<debug::VariableOp>(loc, builder.getStringAttr(paramName),
                                      value, Value{});
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Top-Level Item Conversion
//===----------------------------------------------------------------------===//

namespace {
struct RootVisitor : public BaseVisitor {
  using BaseVisitor::BaseVisitor;
  using BaseVisitor::visit;

  // Handle packages.
  LogicalResult visit(const slang::ast::PackageSymbol &package) {
    return context.convertPackage(package);
  }

  // Handle functions and tasks.
  LogicalResult visit(const slang::ast::SubroutineSymbol &subroutine) {
    return context.convertFunction(subroutine);
  }

  // Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported construct: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Package Conversion
//===----------------------------------------------------------------------===//

namespace {
struct PackageVisitor : public BaseVisitor {
  using BaseVisitor::BaseVisitor;
  using BaseVisitor::visit;

  // Handle functions and tasks.
  LogicalResult visit(const slang::ast::SubroutineSymbol &subroutine) {
    return context.convertFunction(subroutine);
  }

  /// Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported package member: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Module Conversion
//===----------------------------------------------------------------------===//

static moore::ProcedureKind
convertProcedureKind(slang::ast::ProceduralBlockKind kind) {
  switch (kind) {
  case slang::ast::ProceduralBlockKind::Always:
    return moore::ProcedureKind::Always;
  case slang::ast::ProceduralBlockKind::AlwaysComb:
    return moore::ProcedureKind::AlwaysComb;
  case slang::ast::ProceduralBlockKind::AlwaysLatch:
    return moore::ProcedureKind::AlwaysLatch;
  case slang::ast::ProceduralBlockKind::AlwaysFF:
    return moore::ProcedureKind::AlwaysFF;
  case slang::ast::ProceduralBlockKind::Initial:
    return moore::ProcedureKind::Initial;
  case slang::ast::ProceduralBlockKind::Final:
    return moore::ProcedureKind::Final;
  }
  llvm_unreachable("all procedure kinds handled");
}

static moore::NetKind convertNetKind(slang::ast::NetType::NetKind kind) {
  switch (kind) {
  case slang::ast::NetType::Supply0:
    return moore::NetKind::Supply0;
  case slang::ast::NetType::Supply1:
    return moore::NetKind::Supply1;
  case slang::ast::NetType::Tri:
    return moore::NetKind::Tri;
  case slang::ast::NetType::TriAnd:
    return moore::NetKind::TriAnd;
  case slang::ast::NetType::TriOr:
    return moore::NetKind::TriOr;
  case slang::ast::NetType::TriReg:
    return moore::NetKind::TriReg;
  case slang::ast::NetType::Tri0:
    return moore::NetKind::Tri0;
  case slang::ast::NetType::Tri1:
    return moore::NetKind::Tri1;
  case slang::ast::NetType::UWire:
    return moore::NetKind::UWire;
  case slang::ast::NetType::Wire:
    return moore::NetKind::Wire;
  case slang::ast::NetType::WAnd:
    return moore::NetKind::WAnd;
  case slang::ast::NetType::WOr:
    return moore::NetKind::WOr;
  case slang::ast::NetType::Interconnect:
    return moore::NetKind::Interconnect;
  case slang::ast::NetType::UserDefined:
    return moore::NetKind::UserDefined;
  case slang::ast::NetType::Unknown:
    return moore::NetKind::Unknown;
  }
  llvm_unreachable("all net kinds handled");
}

namespace {
struct ModuleVisitor : public BaseVisitor {
  using BaseVisitor::BaseVisitor;
  using BaseVisitor::visit;

  // Skip ports which are already handled by the module itself.
  LogicalResult visit(const slang::ast::PortSymbol &) { return success(); }
  LogicalResult visit(const slang::ast::MultiPortSymbol &) { return success(); }

  // Skip genvars.
  LogicalResult visit(const slang::ast::GenvarSymbol &genvarNode) {
    return success();
  }

  // Skip defparams which have been handled by slang.
  LogicalResult visit(const slang::ast::DefParamSymbol &) { return success(); }

  // Ignore type parameters. These have already been handled by Slang's type
  // checking.
  LogicalResult visit(const slang::ast::TypeParameterSymbol &) {
    return success();
  }

  // Handle instances.
  LogicalResult visit(const slang::ast::InstanceSymbol &instNode) {
    using slang::ast::ArgumentDirection;
    using slang::ast::AssignmentExpression;
    using slang::ast::MultiPortSymbol;
    using slang::ast::PortSymbol;

    auto *moduleLowering = context.convertModuleHeader(&instNode.body);
    if (!moduleLowering)
      return failure();
    auto module = moduleLowering->op;
    auto moduleType = module.getModuleType();

    // Set visibility attribute for instantiated module.
    SymbolTable::setSymbolVisibility(module, SymbolTable::Visibility::Private);

    // Prepare the values that are involved in port connections. This creates
    // rvalues for input ports and appropriate lvalues for output, inout, and
    // ref ports. We also separate multi-ports into the individual underlying
    // ports with their corresponding connection.
    SmallDenseMap<const PortSymbol *, Value> portValues;
    portValues.reserve(moduleType.getNumPorts());

    for (const auto *con : instNode.getPortConnections()) {
      const auto *expr = con->getExpression();

      // Handle unconnected behavior. The expression is null if it have no
      // connection for the port.
      if (!expr) {
        auto *port = con->port.as_if<PortSymbol>();
        if (auto *existingPort =
                moduleLowering->portsBySyntaxNode.lookup(port->getSyntax()))
          port = existingPort;

        switch (port->direction) {
        case ArgumentDirection::In: {
          auto refType = moore::RefType::get(
              cast<moore::UnpackedType>(context.convertType(port->getType())));

          if (const auto *net =
                  port->internalSymbol->as_if<slang::ast::NetSymbol>()) {
            auto netOp = builder.create<moore::NetOp>(
                loc, refType, StringAttr::get(builder.getContext(), net->name),
                convertNetKind(net->netType.netKind), nullptr);
            auto readOp = builder.create<moore::ReadOp>(loc, netOp);
            portValues.insert({port, readOp});
          } else if (const auto *var =
                         port->internalSymbol
                             ->as_if<slang::ast::VariableSymbol>()) {
            auto varOp = builder.create<moore::VariableOp>(
                loc, refType, StringAttr::get(builder.getContext(), var->name),
                nullptr);
            auto readOp = builder.create<moore::ReadOp>(loc, varOp);
            portValues.insert({port, readOp});
          } else {
            return mlir::emitError(loc)
                   << "unsupported internal symbol for unconnected port `"
                   << port->name << "`";
          }
          continue;
        }

        // No need to express unconnected behavior for output port, skip to the
        // next iteration of the loop.
        case ArgumentDirection::Out:
          continue;

        // TODO: Mark Inout port as unsupported and it will be supported later.
        default:
          return mlir::emitError(loc)
                 << "unsupported port `" << port->name << "` ("
                 << slang::ast::toString(port->kind) << ")";
        }
      }

      // Unpack the `<expr> = EmptyArgument` pattern emitted by Slang for
      // output and inout ports.
      if (const auto *assign = expr->as_if<AssignmentExpression>())
        expr = &assign->left();

      // Regular ports lower the connected expression to an lvalue or rvalue and
      // either attach it to the instance as an operand (for input, inout, and
      // ref ports), or assign an instance output to it (for output ports).
      if (auto *port = con->port.as_if<PortSymbol>()) {
        // Convert as rvalue for inputs, lvalue for all others.
        auto value = (port->direction == ArgumentDirection::In)
                         ? context.convertRvalueExpression(*expr)
                         : context.convertLvalueExpression(*expr);
        if (!value)
          return failure();
        if (auto *existingPort =
                moduleLowering->portsBySyntaxNode.lookup(con->port.getSyntax()))
          port = existingPort;
        portValues.insert({port, value});
        continue;
      }

      // Multi-ports lower the connected expression to an lvalue and then slice
      // it up into multiple sub-values, one for each of the ports in the
      // multi-port.
      if (const auto *multiPort = con->port.as_if<MultiPortSymbol>()) {
        // Convert as lvalue.
        auto value = context.convertLvalueExpression(*expr);
        if (!value)
          return failure();
        unsigned offset = 0;
        for (const auto *port : llvm::reverse(multiPort->ports)) {
          if (auto *existingPort = moduleLowering->portsBySyntaxNode.lookup(
                  con->port.getSyntax()))
            port = existingPort;
          unsigned width = port->getType().getBitWidth();
          auto sliceType = context.convertType(port->getType());
          if (!sliceType)
            return failure();
          Value slice = builder.create<moore::ExtractRefOp>(
              loc, moore::RefType::get(cast<moore::UnpackedType>(sliceType)),
              value, offset);
          // Create the "ReadOp" for input ports.
          if (port->direction == ArgumentDirection::In)
            slice = builder.create<moore::ReadOp>(loc, slice);
          portValues.insert({port, slice});
          offset += width;
        }
        continue;
      }

      mlir::emitError(loc) << "unsupported instance port `" << con->port.name
                           << "` (" << slang::ast::toString(con->port.kind)
                           << ")";
      return failure();
    }

    // Match the module's ports up with the port values determined above.
    SmallVector<Value> inputValues;
    SmallVector<Value> outputValues;
    inputValues.reserve(moduleType.getNumInputs());
    outputValues.reserve(moduleType.getNumOutputs());

    for (auto &port : moduleLowering->ports) {
      auto value = portValues.lookup(&port.ast);
      if (port.ast.direction == ArgumentDirection::Out)
        outputValues.push_back(value);
      else
        inputValues.push_back(value);
    }

    // Insert conversions for input ports.
    for (auto [value, type] :
         llvm::zip(inputValues, moduleType.getInputTypes()))
      if (value.getType() != type)
        value =
            builder.create<moore::ConversionOp>(value.getLoc(), type, value);

    // Here we use the hierarchical value recorded in `Context::valueSymbols`.
    // Then we pass it as the input port with the ref<T> type of the instance.
    for (const auto &hierPath : context.hierPaths[&instNode.body])
      if (auto hierValue = context.valueSymbols.lookup(hierPath.valueSym);
          hierPath.hierName && hierPath.direction == ArgumentDirection::In)
        inputValues.push_back(hierValue);

    // Create the instance op itself.
    auto inputNames = builder.getArrayAttr(moduleType.getInputNames());
    auto outputNames = builder.getArrayAttr(moduleType.getOutputNames());
    auto inst = builder.create<moore::InstanceOp>(
        loc, moduleType.getOutputTypes(), builder.getStringAttr(instNode.name),
        FlatSymbolRefAttr::get(module.getSymNameAttr()), inputValues,
        inputNames, outputNames);

    // Record instance's results generated by hierarchical names.
    for (const auto &hierPath : context.hierPaths[&instNode.body])
      if (hierPath.idx && hierPath.direction == ArgumentDirection::Out)
        context.valueSymbols.insert(hierPath.valueSym,
                                    inst->getResult(*hierPath.idx));

    // Assign output values from the instance to the connected expression.
    for (auto [lvalue, output] : llvm::zip(outputValues, inst.getOutputs())) {
      if (!lvalue)
        continue;
      Value rvalue = output;
      auto dstType = cast<moore::RefType>(lvalue.getType()).getNestedType();
      if (dstType != rvalue.getType())
        rvalue = builder.create<moore::ConversionOp>(loc, dstType, rvalue);
      builder.create<moore::ContinuousAssignOp>(loc, lvalue, rvalue);
    }

    return success();
  }

  // Handle variables.
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto loweredType = context.convertType(*varNode.getDeclaredType());
    if (!loweredType)
      return failure();

    Value initial;
    if (const auto *init = varNode.getInitializer()) {
      initial = context.convertRvalueExpression(*init, loweredType);
      if (!initial)
        return failure();
    }

    auto varOp = builder.create<moore::VariableOp>(
        loc, moore::RefType::get(cast<moore::UnpackedType>(loweredType)),
        builder.getStringAttr(varNode.name), initial);
    context.valueSymbols.insert(&varNode, varOp);
    return success();
  }

  // Handle nets.
  LogicalResult visit(const slang::ast::NetSymbol &netNode) {
    auto loweredType = context.convertType(*netNode.getDeclaredType());
    if (!loweredType)
      return failure();

    Value assignment;
    if (const auto *init = netNode.getInitializer()) {
      assignment = context.convertRvalueExpression(*init, loweredType);
      if (!assignment)
        return failure();
    }

    auto netkind = convertNetKind(netNode.netType.netKind);
    if (netkind == moore::NetKind::Interconnect ||
        netkind == moore::NetKind::UserDefined ||
        netkind == moore::NetKind::Unknown)
      return mlir::emitError(loc, "unsupported net kind `")
             << netNode.netType.name << "`";

    auto netOp = builder.create<moore::NetOp>(
        loc, moore::RefType::get(cast<moore::UnpackedType>(loweredType)),
        builder.getStringAttr(netNode.name), netkind, assignment);
    context.valueSymbols.insert(&netNode, netOp);
    return success();
  }

  // Handle continuous assignments.
  LogicalResult visit(const slang::ast::ContinuousAssignSymbol &assignNode) {
    if (const auto *delay = assignNode.getDelay()) {
      auto loc = context.convertLocation(delay->sourceRange);
      return mlir::emitError(loc,
                             "delayed continuous assignments not supported");
    }

    const auto &expr =
        assignNode.getAssignment().as<slang::ast::AssignmentExpression>();
    auto lhs = context.convertLvalueExpression(expr.left());
    if (!lhs)
      return failure();

    auto rhs = context.convertRvalueExpression(
        expr.right(), cast<moore::RefType>(lhs.getType()).getNestedType());
    if (!rhs)
      return failure();

    builder.create<moore::ContinuousAssignOp>(loc, lhs, rhs);
    return success();
  }

  // Handle procedures.
  LogicalResult convertProcedure(moore::ProcedureKind kind,
                                 const slang::ast::Statement &body) {
    auto procOp = builder.create<moore::ProcedureOp>(loc, kind);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&procOp.getBody().emplaceBlock());
    Context::ValueSymbolScope scope(context.valueSymbols);
    if (failed(context.convertStatement(body)))
      return failure();
    if (builder.getBlock())
      builder.create<moore::ReturnOp>(loc);
    return success();
  }

  LogicalResult visit(const slang::ast::ProceduralBlockSymbol &procNode) {
    // Detect `always @(*) <stmt>` and convert to `always_comb <stmt>` if
    // requested by the user.
    if (context.options.lowerAlwaysAtStarAsComb) {
      auto *stmt = procNode.getBody().as_if<slang::ast::TimedStatement>();
      if (procNode.procedureKind == slang::ast::ProceduralBlockKind::Always &&
          stmt &&
          stmt->timing.kind == slang::ast::TimingControlKind::ImplicitEvent)
        return convertProcedure(moore::ProcedureKind::AlwaysComb, stmt->stmt);
    }

    return convertProcedure(convertProcedureKind(procNode.procedureKind),
                            procNode.getBody());
  }

  // Handle generate block.
  LogicalResult visit(const slang::ast::GenerateBlockSymbol &genNode) {
    if (!genNode.isUninstantiated) {
      for (auto &member : genNode.members()) {
        if (failed(member.visit(ModuleVisitor(context, loc))))
          return failure();
      }
    }
    return success();
  }

  // Handle generate block array.
  LogicalResult visit(const slang::ast::GenerateBlockArraySymbol &genArrNode) {
    for (const auto *member : genArrNode.entries) {
      if (failed(member->asSymbol().visit(ModuleVisitor(context, loc))))
        return failure();
    }
    return success();
  }

  // Ignore statement block symbols. These get generated by Slang for blocks
  // with variables and other declarations. For example, having an initial
  // procedure with a variable declaration, such as `initial begin int x;
  // end`, will create the procedure with a block and variable declaration as
  // expected, but will also create a `StatementBlockSymbol` with just the
  // variable layout _next to_ the initial procedure.
  LogicalResult visit(const slang::ast::StatementBlockSymbol &) {
    return success();
  }

  // Ignore sequence declarations. The declarations are already evaluated by
  // Slang and are part of an AssertionInstance.
  LogicalResult visit(const slang::ast::SequenceSymbol &seqNode) {
    return success();
  }

  // Ignore property declarations. The declarations are already evaluated by
  // Slang and are part of an AssertionInstance.
  LogicalResult visit(const slang::ast::PropertySymbol &propNode) {
    return success();
  }

  // Handle functions and tasks.
  LogicalResult visit(const slang::ast::SubroutineSymbol &subroutine) {
    return context.convertFunction(subroutine);
  }

  /// Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported module member: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Structure and Hierarchy Conversion
//===----------------------------------------------------------------------===//

/// Convert an entire Slang compilation to MLIR ops. This is the main entry
/// point for the conversion.
LogicalResult Context::convertCompilation() {
  const auto &root = compilation.getRoot();

  // First only to visit the whole AST to collect the hierarchical names without
  // any operation creating.
  for (auto *inst : root.topInstances)
    if (failed(traverseInstanceBody(inst->body)))
      return failure();

  // Visit all top-level declarations in all compilation units. This does not
  // include instantiable constructs like modules, interfaces, and programs,
  // which are listed separately as top instances.
  for (auto *unit : root.compilationUnits) {
    for (const auto &member : unit->members()) {
      auto loc = convertLocation(member.location);
      if (failed(member.visit(RootVisitor(*this, loc))))
        return failure();
    }
  }

  // Prime the root definition worklist by adding all the top-level modules.
  SmallVector<const slang::ast::InstanceSymbol *> topInstances;
  for (auto *inst : root.topInstances)
    if (!convertModuleHeader(&inst->body))
      return failure();

  // Convert all the root module definitions.
  while (!moduleWorklist.empty()) {
    auto *module = moduleWorklist.front();
    moduleWorklist.pop();
    if (failed(convertModuleBody(module)))
      return failure();
  }

  return success();
}

/// Convert a module and its ports to an empty module op in the IR. Also adds
/// the op to the worklist of module bodies to be lowered. This acts like a
/// module "declaration", allowing instances to already refer to a module even
/// before its body has been lowered.
ModuleLowering *
Context::convertModuleHeader(const slang::ast::InstanceBodySymbol *module) {
  using slang::ast::ArgumentDirection;
  using slang::ast::MultiPortSymbol;
  using slang::ast::ParameterSymbol;
  using slang::ast::PortSymbol;
  using slang::ast::TypeParameterSymbol;

  auto parameters = module->parameters;
  bool hasModuleSame = false;
  // If there is already exist a module that has the same name with this
  // module ,has the same parent scope and has the same parameters we can
  // define this module is a duplicate module
  for (auto const &existingModule : modules) {
    if (module->getDeclaringDefinition() ==
        existingModule.getFirst()->getDeclaringDefinition()) {
      auto moduleParameters = existingModule.getFirst()->parameters;
      hasModuleSame = true;
      for (auto it1 = parameters.begin(), it2 = moduleParameters.begin();
           it1 != parameters.end() && it2 != moduleParameters.end();
           it1++, it2++) {
        // Parameters size different
        if (it1 == parameters.end() || it2 == moduleParameters.end()) {
          hasModuleSame = false;
          break;
        }
        const auto *para1 = (*it1)->symbol.as_if<ParameterSymbol>();
        const auto *para2 = (*it2)->symbol.as_if<ParameterSymbol>();
        // Parameters kind different
        if ((para1 == nullptr) ^ (para2 == nullptr)) {
          hasModuleSame = false;
          break;
        }
        // Compare ParameterSymbol
        if (para1 != nullptr) {
          hasModuleSame = para1->getValue() == para2->getValue();
        }
        // Compare TypeParameterSymbol
        if (para1 == nullptr) {
          auto para1Type = convertType(
              (*it1)->symbol.as<TypeParameterSymbol>().getTypeAlias());
          auto para2Type = convertType(
              (*it2)->symbol.as<TypeParameterSymbol>().getTypeAlias());
          hasModuleSame = para1Type == para2Type;
        }
        if (!hasModuleSame)
          break;
      }
      if (hasModuleSame) {
        module = existingModule.first;
        break;
      }
    }
  }

  auto &slot = modules[module];
  if (slot)
    return slot.get();
  slot = std::make_unique<ModuleLowering>();
  auto &lowering = *slot;

  auto loc = convertLocation(module->location);
  OpBuilder::InsertionGuard g(builder);

  // We only support modules for now. Extension to interfaces and programs
  // should be trivial though, since they are essentially the same thing with
  // only minor differences in semantics.
  if (module->getDefinition().definitionKind !=
      slang::ast::DefinitionKind::Module) {
    mlir::emitError(loc) << "unsupported definition: "
                         << module->getDefinition().getKindString();
    return {};
  }

  // Handle the port list.
  auto block = std::make_unique<Block>();
  SmallVector<hw::ModulePort> modulePorts;

  // It's used to tag where a hierarchical name is on the port list.
  unsigned int outputIdx = 0, inputIdx = 0;
  for (auto *symbol : module->getPortList()) {
    auto handlePort = [&](const PortSymbol &port) {
      auto portLoc = convertLocation(port.location);
      auto type = convertType(port.getType());
      if (!type)
        return failure();
      auto portName = builder.getStringAttr(port.name);
      BlockArgument arg;
      if (port.direction == ArgumentDirection::Out) {
        modulePorts.push_back({portName, type, hw::ModulePort::Output});
        outputIdx++;
      } else {
        // Only the ref type wrapper exists for the time being, the net type
        // wrapper for inout may be introduced later if necessary.
        if (port.direction != ArgumentDirection::In)
          type = moore::RefType::get(cast<moore::UnpackedType>(type));
        modulePorts.push_back({portName, type, hw::ModulePort::Input});
        arg = block->addArgument(type, portLoc);
        inputIdx++;
      }
      lowering.ports.push_back({port, portLoc, arg});
      return success();
    };

    if (const auto *port = symbol->as_if<PortSymbol>()) {
      if (failed(handlePort(*port)))
        return {};
    } else if (const auto *multiPort = symbol->as_if<MultiPortSymbol>()) {
      for (auto *port : multiPort->ports)
        if (failed(handlePort(*port)))
          return {};
    } else {
      mlir::emitError(convertLocation(symbol->location))
          << "unsupported module port `" << symbol->name << "` ("
          << slang::ast::toString(symbol->kind) << ")";
      return {};
    }
  }

  // Mapping hierarchical names into the module's ports.
  for (auto &hierPath : hierPaths[module]) {
    auto hierType = convertType(hierPath.valueSym->getType());
    if (!hierType)
      return {};

    if (auto hierName = hierPath.hierName) {
      // The type of all hierarchical names are marked as the "RefType".
      hierType = moore::RefType::get(cast<moore::UnpackedType>(hierType));
      if (hierPath.direction == ArgumentDirection::Out) {
        hierPath.idx = outputIdx++;
        modulePorts.push_back({hierName, hierType, hw::ModulePort::Output});
      } else {
        hierPath.idx = inputIdx++;
        modulePorts.push_back({hierName, hierType, hw::ModulePort::Input});
        auto hierLoc = convertLocation(hierPath.valueSym->location);
        block->addArgument(hierType, hierLoc);
      }
    }
  }
  auto moduleType = hw::ModuleType::get(getContext(), modulePorts);

  // Pick an insertion point for this module according to the source file
  // location.
  auto it = orderedRootOps.upper_bound(module->location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Create an empty module that corresponds to this module.
  auto moduleOp =
      builder.create<moore::SVModuleOp>(loc, module->name, moduleType);
  orderedRootOps.insert(it, {module->location, moduleOp});
  moduleOp.getBodyRegion().push_back(block.release());
  lowering.op = moduleOp;

  // Add the module to the symbol table of the MLIR module, which uniquifies its
  // name as we'd expect.
  symbolTable.insert(moduleOp);

  // Schedule the body to be lowered.
  moduleWorklist.push(module);

  // Map duplicate port by Syntax
  for (const auto &port : lowering.ports)
    lowering.portsBySyntaxNode.insert({port.ast.getSyntax(), &port.ast});

  return &lowering;
}

/// Convert a module's body to the corresponding IR ops. The module op must have
/// already been created earlier through a `convertModuleHeader` call.
LogicalResult
Context::convertModuleBody(const slang::ast::InstanceBodySymbol *module) {
  auto &lowering = *modules[module];
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(lowering.op.getBody());

  ValueSymbolScope scope(valueSymbols);

  // Collect downward hierarchical names. Such as,
  // module SubA; int x = Top.y; endmodule. The "Top" module is the parent of
  // the "SubA", so "Top.y" is the downward hierarchical name.
  for (auto &hierPath : hierPaths[module])
    if (hierPath.direction == slang::ast::ArgumentDirection::In && hierPath.idx)
      valueSymbols.insert(hierPath.valueSym,
                          lowering.op.getBody()->getArgument(*hierPath.idx));

  // Convert the body of the module.
  for (auto &member : module->members()) {
    auto loc = convertLocation(member.location);
    if (failed(member.visit(ModuleVisitor(*this, loc))))
      return failure();
  }

  // Create additional ops to drive input port values onto the corresponding
  // internal variables and nets, and to collect output port values for the
  // terminator.
  SmallVector<Value> outputs;
  for (auto &port : lowering.ports) {
    Value value;
    if (auto *expr = port.ast.getInternalExpr()) {
      value = convertLvalueExpression(*expr);
    } else if (port.ast.internalSymbol) {
      if (const auto *sym =
              port.ast.internalSymbol->as_if<slang::ast::ValueSymbol>())
        value = valueSymbols.lookup(sym);
    }
    if (!value)
      return mlir::emitError(port.loc, "unsupported port: `")
             << port.ast.name
             << "` does not map to an internal symbol or expression";

    // Collect output port values to be returned in the terminator.
    if (port.ast.direction == slang::ast::ArgumentDirection::Out) {
      if (isa<moore::RefType>(value.getType()))
        value = builder.create<moore::ReadOp>(value.getLoc(), value);
      outputs.push_back(value);
      continue;
    }

    // Assign the value coming in through the port to the internal net or symbol
    // of that port.
    Value portArg = port.arg;
    if (port.ast.direction != slang::ast::ArgumentDirection::In)
      portArg = builder.create<moore::ReadOp>(port.loc, port.arg);
    builder.create<moore::ContinuousAssignOp>(port.loc, value, portArg);
  }

  // Ensure the number of operands of this module's terminator and the number of
  // its(the current module) output ports remain consistent.
  for (auto &hierPath : hierPaths[module])
    if (auto hierValue = valueSymbols.lookup(hierPath.valueSym))
      if (hierPath.direction == slang::ast::ArgumentDirection::Out)
        outputs.push_back(hierValue);

  builder.create<moore::OutputOp>(lowering.op.getLoc(), outputs);
  return success();
}

/// Convert a package and its contents.
LogicalResult
Context::convertPackage(const slang::ast::PackageSymbol &package) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(intoModuleOp.getBody());
  ValueSymbolScope scope(valueSymbols);
  for (auto &member : package.members()) {
    auto loc = convertLocation(member.location);
    if (failed(member.visit(PackageVisitor(*this, loc))))
      return failure();
  }
  return success();
}

/// Convert a function and its arguments to a function declaration in the IR.
/// This does not convert the function body.
FunctionLowering *
Context::declareFunction(const slang::ast::SubroutineSymbol &subroutine) {
  using slang::ast::ArgumentDirection;

  // Check if there already is a declaration for this function.
  auto &lowering = functions[&subroutine];
  if (lowering) {
    if (!lowering->op)
      return {};
    return lowering.get();
  }
  lowering = std::make_unique<FunctionLowering>();
  auto loc = convertLocation(subroutine.location);

  // Pick an insertion point for this function according to the source file
  // location.
  OpBuilder::InsertionGuard g(builder);
  auto it = orderedRootOps.upper_bound(subroutine.location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Class methods are currently not supported.
  if (subroutine.thisVar) {
    mlir::emitError(loc) << "unsupported class method";
    return {};
  }

  // Determine the function type.
  SmallVector<Type> inputTypes;
  SmallVector<Type, 1> outputTypes;

  for (const auto *arg : subroutine.getArguments()) {
    auto type = cast<moore::UnpackedType>(convertType(arg->getType()));
    if (!type)
      return {};
    if (arg->direction == ArgumentDirection::In) {
      inputTypes.push_back(type);
    } else {
      inputTypes.push_back(moore::RefType::get(type));
    }
  }

  if (!subroutine.getReturnType().isVoid()) {
    auto type = convertType(subroutine.getReturnType());
    if (!type)
      return {};
    outputTypes.push_back(type);
  }

  auto funcType = FunctionType::get(getContext(), inputTypes, outputTypes);

  // Prefix the function name with the surrounding namespace to create somewhat
  // sane names in the IR.
  SmallString<64> funcName;
  guessNamespacePrefix(subroutine.getParentScope()->asSymbol(), funcName);
  funcName += subroutine.name;

  // Create a function declaration.
  auto funcOp = builder.create<mlir::func::FuncOp>(loc, funcName, funcType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  orderedRootOps.insert(it, {subroutine.location, funcOp});
  lowering->op = funcOp;

  // Add the function to the symbol table of the MLIR module, which uniquifies
  // its name.
  symbolTable.insert(funcOp);

  return lowering.get();
}

/// Convert a function.
LogicalResult
Context::convertFunction(const slang::ast::SubroutineSymbol &subroutine) {
  // First get or create the function declaration.
  auto *lowering = declareFunction(subroutine);
  if (!lowering)
    return failure();
  ValueSymbolScope scope(valueSymbols);

  // Create a function body block and populate it with block arguments.
  SmallVector<moore::VariableOp> argVariables;
  auto &block = lowering->op.getBody().emplaceBlock();
  for (auto [astArg, type] :
       llvm::zip(subroutine.getArguments(),
                 lowering->op.getFunctionType().getInputs())) {
    auto loc = convertLocation(astArg->location);
    auto blockArg = block.addArgument(type, loc);

    if (isa<moore::RefType>(type)) {
      valueSymbols.insert(astArg, blockArg);
    } else {
      // Convert the body of the function.
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&block);

      auto shadowArg = builder.create<moore::VariableOp>(
          loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
          StringAttr{}, blockArg);
      valueSymbols.insert(astArg, shadowArg);
      argVariables.push_back(shadowArg);
    }
  }

  // Convert the body of the function.
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&block);

  Value returnVar;
  if (subroutine.returnValVar) {
    auto type = convertType(*subroutine.returnValVar->getDeclaredType());
    if (!type)
      return failure();
    returnVar = builder.create<moore::VariableOp>(
        lowering->op.getLoc(),
        moore::RefType::get(cast<moore::UnpackedType>(type)), StringAttr{},
        Value{});
    valueSymbols.insert(subroutine.returnValVar, returnVar);
  }

  if (failed(convertStatement(subroutine.getBody())))
    return failure();

  // If there was no explicit return statement provided by the user, insert a
  // default one.
  if (builder.getBlock()) {
    if (returnVar && !subroutine.getReturnType().isVoid()) {
      Value read = builder.create<moore::ReadOp>(returnVar.getLoc(), returnVar);
      builder.create<mlir::func::ReturnOp>(lowering->op.getLoc(), read);
    } else {
      builder.create<mlir::func::ReturnOp>(lowering->op.getLoc(), ValueRange{});
    }
  }
  if (returnVar && returnVar.use_empty())
    returnVar.getDefiningOp()->erase();

  for (auto var : argVariables) {
    if (llvm::all_of(var->getUsers(),
                     [](auto *user) { return isa<moore::ReadOp>(user); })) {
      for (auto *user : llvm::make_early_inc_range(var->getUsers())) {
        user->getResult(0).replaceAllUsesWith(var.getInitial());
        user->erase();
      }
      var->erase();
    }
  }
  return success();
}
