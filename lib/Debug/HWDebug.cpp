#define GET_OP_CLASSES

#include <memory>

#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVVisitors.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"

mlir::StringRef getSymOpName(mlir::Operation *symOp);
mlir::StringRef getPortVerilogName(mlir::Operation *module,
                                   circt::hw::PortInfo port);

namespace circt::hw {
mlir::StringAttr getVerilogModuleNameAttr(mlir::Operation *module);
} // namespace circt::hw

namespace circt::ExportVerilog {
bool isVerilogExpression(Operation *op);
} // namespace circt::ExportVerilog

namespace circt::debug {

struct HWDebugScope;
class HWDebugContext;

void setEntryLocation(HWDebugScope &scope, const mlir::Location &location,
                      mlir::Operation *op = nullptr);

enum class HWDebugScopeType { None, Assign, Declare, Module, Block };

mlir::StringRef toString(HWDebugScopeType type) {
  switch (type) {
  case HWDebugScopeType::None:
    return "none";
  case HWDebugScopeType::Assign:
    return "assign";
  case HWDebugScopeType::Declare:
    return "decl";
  case HWDebugScopeType::Module:
    return "module";
  case HWDebugScopeType::Block:
    return "block";
  default:
    llvm_unreachable("unknown type");
  }
}

struct HWDebugScope {
public:
  explicit HWDebugScope(HWDebugContext &context, mlir::Operation *op)
      : context(context), op(op) {}

  llvm::SmallVector<HWDebugScope *> scopes;

  mlir::StringRef filename;
  uint32_t line = 0;
  uint32_t column = 0;
  mlir::StringRef condition;

  HWDebugScope *parent = nullptr;

  HWDebugContext &context;

  mlir::Operation *op;

  // NOLINTNEXTLINE
  [[nodiscard]] virtual llvm::json::Value toJSON() const {
    auto res = getScopeJSON(type() == HWDebugScopeType::Block);
    return res;
  }

  [[nodiscard]] virtual HWDebugScopeType type() const {
    return scopes.empty() ? HWDebugScopeType::None : HWDebugScopeType::Block;
  }

protected:
  // NOLINTNEXTLINE
  [[nodiscard]] llvm::json::Object getScopeJSON(bool includeScope) const {
    llvm::json::Object res{{"line", line}};
    if (column > 0) {
      res["column"] = column;
    }
    res["type"] = toString(type());
    if (includeScope) {
      setScope(res);
    }
    if (type() == HWDebugScopeType::Block && !filename.empty()) {
      res["filename"] = filename;
    }
    if (!condition.empty()) {
      res["condition"] = condition;
    }
    return res;
  }

  // NOLINTNEXTLINE
  void setScope(llvm::json::Object &obj) const {
    llvm::json::Array array;
    array.reserve(scopes.size());
    for (auto const *scope : scopes) {
      if (scope)
        array.emplace_back(std::move(scope->toJSON()));
    }
    obj["scope"] = std::move(array);
  }
};

struct HWDebugLineInfo : HWDebugScope {
  enum class LineType {
    None = static_cast<int>(HWDebugScopeType::None),
    Assign = static_cast<int>(HWDebugScopeType::Assign),
    Declare = static_cast<int>(HWDebugScopeType::Declare),
  };

  LineType lineType;

  HWDebugLineInfo(HWDebugContext &context, LineType type, mlir::Operation *op)
      : HWDebugScope(context, op), lineType(type) {}

  [[nodiscard]] llvm::json::Value toJSON() const override {
    auto res = getScopeJSON(false);
    if (!condition.empty()) {
      res["condition"] = condition;
    }
    return res;
  }

  [[nodiscard]] HWDebugScopeType type() const override {
    return static_cast<HWDebugScopeType>(lineType);
  }
};

struct HWDebugVarDef {
  mlir::StringRef name;
  mlir::StringRef value;
  // for how it's always RTL value
  bool rtl = true;

  [[nodiscard]] llvm::json::Value toJSON() const {
    return llvm::json::Object({{"name", name}, {"value", value}, {"rtl", rtl}});
  }
};

struct HWModuleInfo : public HWDebugScope {
public:
  // module names
  mlir::StringRef name;

  llvm::SmallVector<HWDebugVarDef> variables;
  llvm::DenseMap<mlir::StringRef, mlir::StringRef> instances;

  explicit HWModuleInfo(HWDebugContext &context,
                        circt::hw::HWModuleOp *moduleOp)
      : HWDebugScope(context, moduleOp->getOperation()) {
    // index the port names by value
    auto portInfo = moduleOp->getAllPorts();
    for (auto &port : portInfo) {
      StringRef n = getPortVerilogName(*moduleOp, port);
      mlir::Value value;
      if (!port.isOutput()) {
        value = moduleOp->getArgument(port.argNum);
        portNames[value] = n;
      }
    }
  }

  [[nodiscard]] llvm::json::Value toJSON() const override {
    auto res = getScopeJSON(true);
    res["name"] = name;

    llvm::json::Array vars;
    vars.reserve(variables.size());
    for (auto const &varDef : variables) {
      vars.emplace_back(varDef.toJSON());
    }
    res["variables"] = std::move(vars);

    if (!instances.empty()) {
      llvm::json::Array insts;
      insts.reserve(instances.size());
      for (auto const &[n, def] : instances) {
        insts.emplace_back(llvm::json::Object{{"name", n}, {"module", def}});
      }
      res["instances"] = std::move(insts);
    }

    return res;
  }

  [[nodiscard]] HWDebugScopeType type() const override {
    return HWDebugScopeType::Module;
  }

  mlir::StringRef getPortName(mlir::Value value) const {
    return portNames.lookup(value);
  }

private:
  llvm::DenseMap<mlir::Value, mlir::StringRef> portNames;
};

struct HWDebugVarDeclareLineInfo : public HWDebugLineInfo {
  HWDebugVarDeclareLineInfo(HWDebugContext &context, mlir::Operation *op)
      : HWDebugLineInfo(context, LineType::Declare, op) {}

  HWDebugVarDef variable;

  [[nodiscard]] llvm::json::Value toJSON() const override {
    auto res = HWDebugLineInfo::toJSON();
    (*res.getAsObject())["variable"] = std::move(variable.toJSON());
    return res;
  }
};

struct HWDebugVarAssignLineInfo : public HWDebugLineInfo {
  // This also encodes mapping information
  HWDebugVarAssignLineInfo(HWDebugContext &context, mlir::Operation *op)
      : HWDebugLineInfo(context, LineType::Assign, op) {}

  HWDebugVarDef variable;

  [[nodiscard]] llvm::json::Value toJSON() const override {
    auto res = HWDebugLineInfo::toJSON();
    (*res.getAsObject())["variable"] = std::move(variable.toJSON());
    return res;
  }
};

mlir::StringRef findTop(const llvm::SmallVector<HWModuleInfo *> &modules) {
  llvm::SmallDenseSet<mlir::StringRef> names;
  for (auto const *mod : modules) {
    names.insert(mod->name);
  }
  for (auto const *m : modules) {
    for (auto const &[_, name] : m->instances) {
      names.erase(name);
    }
  }
  assert(names.size() == 1);
  return *names.begin();
}

class HWDebugContext {
public:
  [[nodiscard]] llvm::json::Value toJSON() const {
    llvm::json::Object res{{"generator", "circt"}};
    llvm::json::Array array;
    array.reserve(modules.size());
    for (auto const *module : modules) {
      array.emplace_back(std::move(module->toJSON()));
    }
    res["table"] = std::move(array);
    auto top = findTop(modules);
    res["top"] = top;
    return res;
  }

  template <typename T, typename... Args>
  T *createScope(Args &&...args) {
    auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
    auto *res = ptr.get();
    scopes.emplace_back(std::move(ptr));
    if constexpr (std::is_same<T, HWModuleInfo>::value) {
      modules.emplace_back(res);
    }
    return res;
  }

  HWDebugScope *getNormalScope(::mlir::Operation *op) {
    auto *ptr = createScope<HWDebugScope>(*this, op);
    if (op)
      setEntryLocation(*ptr, op->getLoc());
    return ptr;
  }

  [[nodiscard]] const llvm::SmallVector<HWModuleInfo *> &getModules() const {
    return modules;
  }

private:
  llvm::SmallVector<HWModuleInfo *> modules;
  llvm::SmallVector<std::unique_ptr<HWDebugScope>> scopes;
};

void setEntryLocation(HWDebugScope &scope, const mlir::Location &location,
                      mlir::Operation *op) {
  // need to get the containing module, as well as the line number
  // information
  auto const fileLoc = location.cast<::mlir::FileLineColLoc>();
  auto const line = fileLoc.getLine();
  auto const column = fileLoc.getColumn();

  scope.line = line;
  scope.column = column;
}

class HWDebugBuilder {
public:
  HWDebugBuilder(HWDebugContext &context) : context(context) {}

  HWDebugVarDeclareLineInfo *createVarDeclaration(::mlir::Value value) {
    auto loc = value.getLoc();
    auto *op = value.getDefiningOp();
    auto *targetOp = getDebugOp(value, op);
    if (!targetOp)
      return nullptr;

    // need to get the containing module, as well as the line number
    // information
    auto *info = context.createScope<HWDebugVarDeclareLineInfo>(context, op);
    setEntryLocation(*info, loc);
    info->variable = createVarDef(targetOp);
    return info;
  }

  HWDebugVarAssignLineInfo *createAssign(::mlir::Value value,
                                         ::mlir::Operation *op) {
    // only create assign if the target has frontend variable
    auto *targetOp = getDebugOp(value, op);
    if (!targetOp)
      return nullptr;

    auto loc = op->getLoc();

    auto *assign = context.createScope<HWDebugVarAssignLineInfo>(context, op);
    setEntryLocation(*assign, loc);

    assign->variable = createVarDef(targetOp);

    return assign;
  }

  HWDebugVarDef createVarDef(::mlir::Operation *op) {
    // the OP has to have this attr. need to check before calling this function
    auto frontEndName =
        op->getAttr("hw.debug.name").cast<mlir::StringAttr>().strref();
    auto rtlName = ::getSymOpName(op);
    HWDebugVarDef var{.name = frontEndName, .value = rtlName, .rtl = true};
    return var;
  }

  HWModuleInfo *createModule(circt::hw::HWModuleOp *op) {
    auto *info = context.createScope<HWModuleInfo>(context, op);
    setEntryLocation(*info, op->getLoc(), op->getOperation());
    return info;
  }

  HWDebugScope *createScope(::mlir::Operation *op,
                            HWDebugScope *parent = nullptr) {
    auto *res = context.getNormalScope(op);
    if (parent) {
      res->parent = parent;
      parent->scopes.emplace_back(res);
    }
    return res;
  }

private:
  HWDebugContext &context;

  static mlir::Operation *getDebugOp(mlir::Value value, mlir::Operation *op) {
    auto *valueOP = value.getDefiningOp();
    if (valueOP && valueOP->hasAttr("hw.debug.name")) {
      return valueOP;
    }
    if (op && op->hasAttr("hw.debug.name")) {
      return op;
    }
    return nullptr;
  }
};

// hgdb only supports a subsets of operations, nor does it support sign
// conversion. if the expression is complex we need to insert another pass that
// generate a temp wire that holds the expression and use that in hgdb instead.
// for now this is not an issue since Chisel is already doing this (somewhat)
class DebugExprPrinter
    : public circt::comb::CombinationalVisitor<DebugExprPrinter>,
      public circt::hw::TypeOpVisitor<DebugExprPrinter> {
public:
  explicit DebugExprPrinter(llvm::raw_ostream &os, HWModuleInfo *module)
      : os(os), module(module) {}

  void printExpr(mlir::Value value) {
    auto *op = value.getDefiningOp();
    if (op && ExportVerilog::isVerilogExpression(op)) {
      os << getSymOpName(op);
      return;
    }
    if (op) {
      dispatchCombinationalVisitor(op);
    } else {
      auto ref = module->getPortName(value);
      os << ref;
    }
  }
  // comb ops
  // supported
  void visitComb(circt::comb::AddOp op) { visitBinary(op, "+"); }
  void visitComb(circt::comb::SubOp op) { visitBinary(op, "-"); }
  void visitComb(circt::comb::MulOp op) {
    assert(op.getNumOperands() == 2 && "prelowering should handle variadics");
    return visitBinary(op, "*");
  }
  void visitComb(circt::comb::DivUOp op) { visitBinary(op, "/"); }
  void visitComb(circt::comb::DivSOp op) { visitBinary(op, "/"); }
  void visitComb(circt::comb::ModUOp op) { visitBinary(op, "%"); }
  void visitComb(circt::comb::ModSOp op) { visitBinary(op, "%"); }

  void visitComb(circt::comb::AndOp op) { visitBinary(op, "&"); }
  void visitComb(circt::comb::OrOp op) { visitBinary(op, "|"); }
  void visitComb(circt::comb::XorOp op) {
    if (op.isBinaryNot())
      visitUnary(op, "~");
    visitBinary(op, "^");
  }

  void visitComb(circt::comb::ICmpOp op) {
    // this is copied from ExportVerilog.cpp. probably need some
    // code refactoring since this is duplicated logic
    const char *symop[] = {"==", "!=", "<",  "<=", ">",
                           ">=", "<",  "<=", ">",  ">="};
    auto pred = static_cast<uint64_t>(op.predicate());
    visitBinary(op, symop[pred]);
  }

  // unsupported
  void visitComb(circt::comb::MuxOp op) { visitUnsupported(op); }
  void visitComb(circt::comb::ShlOp op) { visitUnsupported(op); }
  void visitComb(circt::comb::ShrUOp op) { visitUnsupported(op); }
  void visitComb(circt::comb::ShrSOp op) { visitUnsupported(op); }

  void visitComb(circt::comb::ParityOp op) { visitUnsupported(op); }

  void visitComb(circt::comb::ReplicateOp op) { visitUnsupported(op); }
  void visitComb(circt::comb::ConcatOp op) { visitUnsupported(op); }
  void visitComb(circt::comb::ExtractOp op) { visitUnsupported(op); }

  // type ops
  void visitTypeOp(circt::hw::ConstantOp op) { os << op.value(); }
  void visitTypeOp(circt::hw::ParamValueOp op) { os << op.value(); }
  // unsupported
  void visitTypeOp(circt::hw::BitcastOp op) { visitUnsupported(op); }
  void visitTypeOp(circt::hw::ArraySliceOp op) { visitUnsupported(op); }
  void visitTypeOp(circt::hw::ArrayGetOp op) { visitUnsupported(op); }
  void visitTypeOp(circt::hw::ArrayCreateOp op) { visitUnsupported(op); }
  void visitTypeOp(circt::hw::ArrayConcatOp op) { visitUnsupported(op); }
  void visitTypeOp(circt::hw::StructCreateOp op) { visitUnsupported(op); }
  void visitTypeOp(circt::hw::StructExtractOp op) { visitUnsupported(op); }
  void visitTypeOp(circt::hw::StructInjectOp op) { visitUnsupported(op); }

  void visitBinary(mlir::Operation *op, mlir::StringRef opStr) {
    // always emit paraphrases
    os << '(';
    auto left = op->getOperand(0);
    printExpr(left);
    os << ' ' << opStr << ' ';
    auto right = op->getOperand(1);
    printExpr(right);
    os << ')';
  }

  void visitUnary(mlir::Operation *op, mlir::StringRef opStr) {
    // always emit paraphrases
    os << '(';
    os << opStr;
    auto target = op->getOperand(0);
    printExpr(target);
    os << ')';
  }

private:
  llvm::raw_ostream &os;
  HWModuleInfo *module;

  static void visitUnsupported(mlir::Operation *op) {
    op->emitError("Unsupported op in debug expression");
  }
};

class DebugStmtVisitor : public circt::hw::StmtVisitor<DebugStmtVisitor>,
                         public circt::sv::Visitor<DebugStmtVisitor, void> {
public:
  DebugStmtVisitor(HWDebugBuilder &builder, HWModuleInfo *module)
      : builder(builder), module(module), currentScope(module) {}

  void visitStmt(circt::hw::InstanceOp op) {
    auto instNameRef = ::getSymOpName(op);
    // need to find definition names
    auto *mod = op.getReferencedModule(nullptr);
    auto moduleNameStr = circt::hw::getVerilogModuleNameAttr(mod).strref();
    module->instances.insert(std::make_pair(instNameRef, moduleNameStr));
  }

  void visitSV(circt::sv::RegOp op) {
    // we treat this as a generator variable
    // only generate if we have annotated in the frontend
    if (hasDebug(op)) {
      auto var = builder.createVarDef(op);
      module->variables.emplace_back(var);
    }
  }

  void visitSV(circt::sv::WireOp op) {
    if (hasDebug(op)) {
      auto var = builder.createVarDef(op);
      module->variables.emplace_back(var);
    }
  }

  // assignment
  // we only care about the target of the assignment
  void visitSV(circt::sv::AssignOp op) {
    if (hasDebug(op)) {
      auto target = op.dest();
      auto *assign = builder.createAssign(target, op);
      if (assign)
        currentScope->scopes.emplace_back(assign);
    }
  }

  void visitSV(circt::sv::BPAssignOp op) {
    if (hasDebug(op)) {
      auto target = op.dest();
      auto *assign = builder.createAssign(target, op);
      if (assign)
        currentScope->scopes.emplace_back(assign);
    }
  }

  void visitSV(circt::sv::PAssignOp op) {
    if (hasDebug(op)) {
      auto target = op.dest();
      auto *assign = builder.createAssign(target, op);
      if (assign)
        currentScope->scopes.emplace_back(assign);
    }
  }

  void visitStmt(circt::hw::OutputOp op) {
    hw::HWModuleOp parent = op->getParentOfType<hw::HWModuleOp>();
    for (auto i = 0u; i < parent.getPorts().outputs.size(); i++) {
      auto operand = op.getOperand(i);
      auto *assign = builder.createAssign(operand, op);
      if (assign)
        currentScope->scopes.emplace_back(assign);
    }
  }

  // visit blocks
  void visitSV(circt::sv::AlwaysOp op) {
    // creating a scope
    auto *scope = builder.createScope(op, currentScope);
    auto *temp = currentScope;
    currentScope = scope;
    visitBlock(*op.getBodyBlock());
    currentScope = temp;
  }

  void visitSV(circt::sv::AlwaysCombOp op) { // creating a scope
    auto *scope = builder.createScope(op, currentScope);
    auto *temp = currentScope;
    currentScope = scope;
    visitBlock(*op.getBodyBlock());
    currentScope = temp;
  }

  void visitSV(circt::sv::AlwaysFFOp op) {
    if (op.getResetBlock()) {
      // creating a scope
      auto *scope = builder.createScope(op, currentScope);
      auto *temp = currentScope;
      currentScope = scope;
      visitBlock(*op.getResetBlock());
      currentScope = temp;
    }
    {
      // creating a scope
      auto *scope = builder.createScope(op, currentScope);
      auto *temp = currentScope;
      currentScope = scope;
      visitBlock(*op.getBodyBlock());
      currentScope = temp;
    }
  }

  void visitSV(circt::sv::InitialOp op) { // creating a scope
    auto *scope = builder.createScope(op, currentScope);
    auto *temp = currentScope;
    currentScope = scope;
    visitBlock(*op.getBodyBlock());
    currentScope = temp;
  }

  void visitSV(circt::sv::IfOp op) {
    // first, the statement itself is a line
    builder.createScope(op, currentScope);
    auto cond = getCondString(op.cond());
    if (cond.empty()) {
      op->emitError("Unsupported if statement condition");
      return;
    }
    if (auto *body = op.getThenBlock()) {
      // true
      if (!body->empty()) {
        auto *trueBlock = builder.createScope(op, currentScope);
        auto *temp = currentScope;
        currentScope = trueBlock;
        trueBlock->condition = mlir::StringAttr::get(op.getContext(), cond);
        visitBlock(*body);
        currentScope = temp;
      }
    }
    if (auto *elseBody = op.getElseBlock()) {
      // false
      if (!elseBody->empty()) {
        auto *elseBlock = builder.createScope(op, currentScope);
        auto *temp = currentScope;
        currentScope = elseBlock;
        elseBlock->condition =
            mlir::StringAttr::get(op->getContext(), "!" + cond);
        visitBlock(*elseBody);
        currentScope = temp;
      }
    }
  }

  void visitSV(circt::sv::CaseZOp op) {
    auto cond = getCondString(op.cond());
    if (cond.empty()) {
      op->emitError("Unsupported case statement condition");
      return;
    }

    auto addScope = [this](const std::string caseCond, mlir::Block *block) {
      // use nullptr since it's auxiliary
      auto *scope = builder.createScope(nullptr, currentScope);
      auto *temp = currentScope;
      currentScope = scope;
      scope->condition = caseCond;
      visitBlock(*block);
      currentScope = temp;
    };

    mlir::Block *defaultBlock = nullptr;
    llvm::SmallVector<uint64_t> values;
    for (auto caseInfo : op.getCases()) {
      auto pattern = caseInfo.pattern;
      if (pattern.isDefault()) {
        defaultBlock = caseInfo.block;
      } else {
        // Currently, hgdb doesn't support z or ?
        // so we have to turn the pattern into an integer
        auto value = pattern.attr.getUInt();
        auto caseCond = cond + " == " + std::to_string(value);
        addScope(caseCond, caseInfo.block);
        values.emplace_back(value);
      }
    }
    if (defaultBlock) {
      // negate all values
      std::string defaultCond;
      for (auto i = 0u; i < values.size(); i++) {
        defaultCond.append("(" + cond + " != " + std::to_string(values[i]) +
                           ")");
        if (i != (values.size() - 1)) {
          defaultCond.append(" && ");
        }
      }
      addScope(defaultCond, defaultBlock);
    }
  }

  // noop HW visit functions
  void visitStmt(circt::hw::ProbeOp) {}
  void visitStmt(circt::hw::TypedeclOp) {}

  void visitStmt(circt::hw::TypeScopeOp op) {}

  // noop SV visit functions
  void visitSV(circt::sv::ReadInOutOp) {}
  void visitSV(circt::sv::ArrayIndexInOutOp) {}
  void visitSV(circt::sv::VerbatimExprOp) {}
  void visitSV(circt::sv::VerbatimExprSEOp) {}
  void visitSV(circt::sv::IndexedPartSelectInOutOp) {}
  void visitSV(circt::sv::IndexedPartSelectOp) {}
  void visitSV(circt::sv::StructFieldInOutOp) {}
  void visitSV(circt::sv::ConstantXOp) {}
  void visitSV(circt::sv::ConstantZOp) {}
  void visitSV(circt::sv::LocalParamOp) {}
  void visitSV(circt::sv::XMROp) {}
  void visitSV(circt::sv::IfDefOp) {}
  void visitSV(circt::sv::IfDefProceduralOp) {}
  void visitSV(circt::sv::ForceOp) {}
  void visitSV(circt::sv::ReleaseOp) {}
  void visitSV(circt::sv::AliasOp) {}
  void visitSV(circt::sv::FWriteOp) {}
  void visitSV(circt::sv::VerbatimOp) {}
  void visitSV(circt::sv::InterfaceOp) {}
  void visitSV(circt::sv::InterfaceSignalOp) {}
  void visitSV(circt::sv::InterfaceModportOp) {}
  void visitSV(circt::sv::InterfaceInstanceOp) {}
  void visitSV(circt::sv::GetModportOp) {}
  void visitSV(circt::sv::AssignInterfaceSignalOp) {}
  void visitSV(circt::sv::ReadInterfaceSignalOp) {}
  void visitSV(circt::sv::AssertOp) {}
  void visitSV(circt::sv::AssumeOp) {}
  void visitSV(circt::sv::CoverOp) {}
  void visitSV(circt::sv::AssertConcurrentOp) {}
  void visitSV(circt::sv::AssumeConcurrentOp) {}
  void visitSV(circt::sv::CoverConcurrentOp) {}
  void visitSV(circt::sv::BindOp) {}
  void visitSV(circt::sv::StopOp) {}
  void visitSV(circt::sv::FinishOp) {}
  void visitSV(circt::sv::ExitOp) {}
  void visitSV(circt::sv::FatalOp) {}
  void visitSV(circt::sv::ErrorOp) {}
  void visitSV(circt::sv::WarningOp) {}
  void visitSV(circt::sv::InfoOp) {}

  // ignore invalid stuff
  void visitInvalidStmt(Operation *) {}
  void visitInvalidSV(Operation *) {}

  void dispatch(mlir::Operation *op) {
    dispatchStmtVisitor(op);
    dispatchSVVisitor(op);
  }

  void visitBlock(mlir::Block &block) {
    for (auto &op : block) {
      dispatch(&op);
    }
  }

private:
  HWDebugBuilder &builder;
  HWModuleInfo *module;
  HWDebugScope *currentScope;

  bool hasDebug(mlir::Operation *op) {
    auto r = op && op->hasAttr("hw.debug.name");
    return r;
  }

  std::string getCondString(mlir::Value value) const {
    std::string cond;
    llvm::raw_string_ostream os(cond);
    DebugExprPrinter p(os, module);
    p.printExpr(value);
    return cond;
  }
};

mlir::StringAttr getFilenameFromScopeOp(HWDebugScope *scope) {
  auto loc = scope->op->getLoc().cast<mlir::FileLineColLoc>();
  return loc.getFilename();
}

// NOLINTNEXTLINE
void setScopeFilename(HWDebugScope *scope, HWDebugBuilder &builder) {
  // assuming the current scope is already fixed
  auto scopeFilename = getFilenameFromScopeOp(scope);
  for (auto i = 0u; i < scope->scopes.size(); i++) {
    auto *entry = scope->scopes[i];
    auto entryFilename = getFilenameFromScopeOp(entry);
    if (entryFilename != scopeFilename) {
      // need to set this entry's filename
      if (entry->type() == HWDebugScopeType::Block) {
        // set its filename
        entry->filename = entryFilename;
      } else {
        // need to create a scope to contain this one
        auto *newScope = builder.createScope(entry->op);
        // set the line to 0 since it's an artificial scope
        newScope->line = 0;
        newScope->filename = entryFilename;
        newScope->scopes.emplace_back(entry);
        entry->parent = newScope;
        newScope->parent = scope;
        scope->scopes[i] = newScope;
      }
    }
  }
  // merge scopes with the same filename
  // we assume at this stage most of the entries are block entry now
  llvm::DenseMap<mlir::StringRef, HWDebugScope *> filenameMapping;
  for (auto i = 0u; i < scope->scopes.size(); i++) {
    auto *entry = scope->scopes[i];
    // we only touch non-existing block, i.e. created for holding actual
    // scopes
    if (entry->type() == HWDebugScopeType::Block && entry->line == 0) {
      auto filename = entry->filename;
      if (filenameMapping.find(filename) == filenameMapping.end()) {
        filenameMapping[filename] = entry;
      } else {
        auto *parent = filenameMapping[filename];
        // merge
        parent->scopes.reserve(entry->scopes.size() + parent->scopes.size());
        for (auto *p : entry->scopes) {
          parent->scopes.emplace_back(p);
          p->parent = p;
        }

        // delete the old entry
        scope->scopes[i] = nullptr;
      }
    }
  }
  // clear the nullptr
  scope->scopes.erase(std::remove_if(scope->scopes.begin(), scope->scopes.end(),
                                     [](auto *p) { return !p; }),
                      scope->scopes.end());

  // recursively setting filenames
  for (auto *entry : scope->scopes) {
    setScopeFilename(entry, builder);
  }
}

void exportDebugTable(mlir::ModuleOp moduleOp, const std::string &filename) {
  // collect all the files
  HWDebugContext context;
  HWDebugBuilder builder(context);
  for (auto &op : *moduleOp.getBody()) {
    mlir::TypeSwitch<mlir::Operation *>(&op).Case<circt::hw::HWModuleOp>(
        [&builder](circt::hw::HWModuleOp mod) {
          // get verilog name
          auto defName = circt::hw::getVerilogModuleNameAttr(mod);
          auto *module = builder.createModule(&mod);
          module->name = defName;
          DebugStmtVisitor visitor(builder, module);
          auto *body = mod.getBodyBlock();
          visitor.visitBlock(*body);
        });
  }
  // fixing filenames
  auto const &modules = context.getModules();
  for (auto *m : modules) {
    setScopeFilename(m, builder);
  }
  auto json = context.toJSON();

  std::error_code error;
  llvm::raw_fd_ostream os(filename, error);
  if (!error) {
    os << json;
  }
  os.close();
}

struct ExportDebugTablePass : public ::mlir::OperationPass<mlir::ModuleOp> {
  ExportDebugTablePass(std::string filename)
      : ::mlir::OperationPass<mlir::ModuleOp>(
            ::mlir::TypeID::get<ExportDebugTablePass>()),
        filename(filename) {}
  ExportDebugTablePass(const ExportDebugTablePass &other)
      : ::mlir::OperationPass<mlir::ModuleOp>(other), filename(other.filename) {
  }

  void runOnOperation() override {
    exportDebugTable(getOperation(), filename);
    markAllAnalysesPreserved();
  }

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("export-hgdb");
  }
  ::llvm::StringRef getArgument() const override { return "export-hgdb"; }

  ::llvm::StringRef getDescription() const override {
    return "Generate symbol table for HGDB";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ExportHGDB");
  }
  ::llvm::StringRef getName() const override { return "ExportHGDB"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<ExportDebugTablePass>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<ExportDebugTablePass>(
        *static_cast<const ExportDebugTablePass *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {

    registry.insert<circt::sv::SVDialect>();

    registry.insert<circt::comb::CombDialect>();

    registry.insert<circt::hw::HWDialect>();
  }

private:
  std::string filename;
};

std::unique_ptr<mlir::Pass> createExportHGDBPass(std::string filename) {
  return std::make_unique<ExportDebugTablePass>(std::move(filename));
}

} // namespace circt::debug