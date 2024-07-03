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
// Module Member Conversion
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
struct MemberVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  MemberVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  // Skip empty members (stray semicolons).
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

  // Skip ports which are already handled by the module itself.
  LogicalResult visit(const slang::ast::PortSymbol &) { return success(); }
  LogicalResult visit(const slang::ast::MultiPortSymbol &) { return success(); }

  // Skip genvars.
  LogicalResult visit(const slang::ast::GenvarSymbol &genvarNode) {
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
        switch (port->direction) {
        case slang::ast::ArgumentDirection::In: {
          auto refType = moore::RefType::get(
              cast<moore::UnpackedType>(context.convertType(port->getType())));

          if (const auto *net =
                  port->internalSymbol->as_if<slang::ast::NetSymbol>()) {
            auto netOp = builder.create<moore::NetOp>(
                loc, refType, StringAttr::get(builder.getContext(), net->name),
                convertNetKind(net->netType.netKind), nullptr);
            auto readOp = builder.create<moore::ReadOp>(
                loc, refType.getNestedType(), netOp);
            portValues.insert({port, readOp});
          } else if (const auto *var =
                         port->internalSymbol
                             ->as_if<slang::ast::VariableSymbol>()) {
            auto varOp = builder.create<moore::VariableOp>(
                loc, refType, StringAttr::get(builder.getContext(), var->name),
                nullptr);
            auto readOp = builder.create<moore::ReadOp>(
                loc, refType.getNestedType(), varOp);
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
        case slang::ast::ArgumentDirection::Out:
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
        auto value = (port->direction == slang::ast::ArgumentDirection::In)
                         ? context.convertRvalueExpression(*expr)
                         : context.convertLvalueExpression(*expr);
        if (!value)
          return failure();
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
        auto i32 = moore::IntType::getInt(context.getContext(), 32);
        for (const auto *port : llvm::reverse(multiPort->ports)) {
          unsigned width = port->getType().getBitWidth();
          auto index = builder.create<moore::ConstantOp>(loc, i32, offset);
          auto sliceType = context.convertType(port->getType());
          if (!sliceType)
            return failure();
          Value slice = builder.create<moore::ExtractRefOp>(
              loc, moore::RefType::get(cast<moore::UnpackedType>(sliceType)),
              value, index);
          // Read to map to rvalue for input ports.
          if (port->direction == slang::ast::ArgumentDirection::In)
            slice = builder.create<moore::ReadOp>(loc, sliceType, slice);
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

    // Create the instance op itself.
    auto inputNames = builder.getArrayAttr(moduleType.getInputNames());
    auto outputNames = builder.getArrayAttr(moduleType.getOutputNames());
    auto inst = builder.create<moore::InstanceOp>(
        loc, moduleType.getOutputTypes(), builder.getStringAttr(instNode.name),
        FlatSymbolRefAttr::get(module.getSymNameAttr()), inputValues,
        inputNames, outputNames);

    // Assign output values from the instance to the connected expression.
    for (auto [lvalue, output] : llvm::zip(outputValues, inst.getOutputs()))
      if (lvalue)
        builder.create<moore::ContinuousAssignOp>(loc, lvalue, output);

    return success();
  }

  // Handle variables.
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto loweredType = context.convertType(*varNode.getDeclaredType());
    if (!loweredType)
      return failure();

    Value initial;
    if (const auto *init = varNode.getInitializer()) {
      initial = context.convertRvalueExpression(*init);
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
    if (netNode.getInitializer()) {
      assignment = context.convertRvalueExpression(*netNode.getInitializer());
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
    auto rhs = context.convertRvalueExpression(expr.right());
    if (!lhs || !rhs)
      return failure();

    builder.create<moore::ContinuousAssignOp>(loc, lhs, rhs);
    return success();
  }

  // Handle procedures.
  LogicalResult visit(const slang::ast::ProceduralBlockSymbol &procNode) {
    auto procOp = builder.create<moore::ProcedureOp>(
        loc, convertProcedureKind(procNode.procedureKind));
    procOp.getBodyRegion().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(procOp.getBody());
    Context::ValueSymbolScope scope(context.valueSymbols);
    return context.convertStatement(procNode.getBody());
  }

  // Handle parameters.
  LogicalResult visit(const slang::ast::ParameterSymbol &paramNode) {
    auto type = cast<moore::IntType>(context.convertType(paramNode.getType()));
    if (!type)
      return failure();

    auto valueInt = paramNode.getValue().integer().as<uint64_t>().value();
    Value value = builder.create<moore::ConstantOp>(loc, type, valueInt);

    auto namedConstantOp = builder.create<moore::NamedConstantOp>(
        loc, type, builder.getStringAttr(paramNode.name),
        paramNode.isLocalParam()
            ? moore::NamedConstAttr::get(context.getContext(),
                                         moore::NamedConst::LocalParameter)
            : moore::NamedConstAttr::get(context.getContext(),
                                         moore::NamedConst::Parameter),
        value);
    context.valueSymbols.insert(&paramNode, namedConstantOp);
    return success();
  }

  // Handle specparam.
  LogicalResult visit(const slang::ast::SpecparamSymbol &spNode) {
    auto type = cast<moore::IntType>(context.convertType(spNode.getType()));
    if (!type)
      return failure();

    auto valueInt = spNode.getValue().integer().as<uint64_t>().value();
    Value value = builder.create<moore::ConstantOp>(loc, type, valueInt);

    auto namedConstantOp = builder.create<moore::NamedConstantOp>(
        loc, type, builder.getStringAttr(spNode.name),
        moore::NamedConstAttr::get(context.getContext(),
                                   moore::NamedConst::SpecParameter),
        value);
    context.valueSymbols.insert(&spNode, namedConstantOp);
    return success();
  }

  // Handle generate block.
  LogicalResult visit(const slang::ast::GenerateBlockSymbol &genNode) {
    if (!genNode.isUninstantiated) {
      for (auto &member : genNode.members()) {
        if (failed(member.visit(MemberVisitor(context, loc))))
          return failure();
      }
    }
    return success();
  }

  // Handle generate block array.
  LogicalResult visit(const slang::ast::GenerateBlockArraySymbol &genArrNode) {
    for (const auto *member : genArrNode.entries) {
      if (failed(member->asSymbol().visit(MemberVisitor(context, loc))))
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

  /// Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported construct: ")
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
LogicalResult
Context::convertCompilation(slang::ast::Compilation &compilation) {
  const auto &root = compilation.getRoot();

  // Visit all top-level declarations in all compilation units. This does not
  // include instantiable constructs like modules, interfaces, and programs,
  // which are listed separately as top instances.
  for (auto *unit : root.compilationUnits) {
    for (const auto &member : unit->members()) {
      // Error out on all top-level declarations.
      auto loc = convertLocation(member.location);
      return mlir::emitError(loc, "unsupported construct: ")
             << slang::ast::toString(member.kind);
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
  using slang::ast::PortSymbol;

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
    mlir::emitError(loc) << "unsupported construct: "
                         << module->getDefinition().getKindString();
    return {};
  }

  // Handle the port list.
  auto block = std::make_unique<Block>();
  SmallVector<hw::ModulePort> modulePorts;
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
      } else {
        // Only the ref type wrapper exists for the time being, the net type
        // wrapper for inout may be introduced later if necessary.
        if (port.direction != slang::ast::ArgumentDirection::In)
          type = moore::RefType::get(cast<moore::UnpackedType>(type));
        modulePorts.push_back({portName, type, hw::ModulePort::Input});
        arg = block->addArgument(type, portLoc);
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
  auto moduleType = hw::ModuleType::get(getContext(), modulePorts);

  // Pick an insertion point for this module according to the source file
  // location.
  auto it = orderedRootOps.lower_bound(module->location);
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
  return &lowering;
}

/// Convert a module's body to the corresponding IR ops. The module op must have
/// already been created earlier through a `convertModuleHeader` call.
LogicalResult
Context::convertModuleBody(const slang::ast::InstanceBodySymbol *module) {
  auto &lowering = *modules[module];
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(lowering.op.getBody());

  // Convert the body of the module.
  ValueSymbolScope scope(valueSymbols);
  for (auto &member : module->members()) {
    auto loc = convertLocation(member.location);
    if (failed(member.visit(MemberVisitor(*this, loc))))
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
        value = builder.create<moore::ReadOp>(
            value.getLoc(),
            cast<moore::RefType>(value.getType()).getNestedType(), value);
      outputs.push_back(value);
      continue;
    }

    // Assign the value coming in through the port to the internal net or symbol
    // of that port.
    Value portArg = port.arg;
    if (port.ast.direction != slang::ast::ArgumentDirection::In)
      portArg = builder.create<moore::ReadOp>(
          port.loc, cast<moore::RefType>(value.getType()).getNestedType(),
          port.arg);
    builder.create<moore::ContinuousAssignOp>(port.loc, value, portArg);
  }
  builder.create<moore::OutputOp>(lowering.op.getLoc(), outputs);

  return success();
}
