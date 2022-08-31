#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "assert.h"
#include <iostream>
#include <variant>

using namespace mlir;
using namespace circt;

/*
CPPCDE: C++ CIRCT Design Entry

Syntactic sugar for creating CIRCT modules and operations. This library is
intended to be used in places where a large amount of CIRCT IR needs to be
generated (think generators) within a pass, wherein a syntactically sugar'ed
approach will lead itself to a significant decrease in the LOC required.

... not a HDL...!
*/

namespace circt {
namespace cppcde {

namespace {
/// Unwraps a range of CDEValues to their underlying mlir::Value's.
template <typename CDEValue>
static llvm::SmallVector<Value> unwrap(llvm::ArrayRef<CDEValue> values) {
  SmallVector<Value> unwrapped;
  llvm::transform(values, std::back_inserter(unwrapped),
                  [](auto &v) { return v.get(); });
  return unwrapped;
}
} // namespace

// Port annotations.
enum class PortKind { Clock, Reset, Default };

// A context defining the operations that define a given semantic.
// This can be used to overload the inferred operations in a CPPCDE
// generator.
struct CDECombOperationSelector {
  // Logical operators.
  using And = comb::AndOp;
  using Or = comb::OrOp;
  using XOr = comb::XorOp;
  using Shl = comb::ShlOp;
  using ShrS = comb::ShrSOp;
  using ShrU = comb::ShrUOp;
  using Concat = comb::ConcatOp;

  // Arithmetic operators.
  using Mul = comb::MulOp;
  using Add = comb::AddOp;
  using Sub = comb::SubOp;
  using DivU = comb::DivUOp;
  using DivS = comb::DivSOp;
  using ModU = comb::ModUOp;
  using ModS = comb::ModSOp;
};

// An object containing the shared state of a generator.
struct CDEState {
  CDEState(Location loc, OpBuilder &b, BackedgeBuilder &bb)
      : loc(loc), b(b), bb(bb) {}
  Location loc;
  OpBuilder &b;
  BackedgeBuilder &bb;
  Value clk;
  Value rst;
};

// Base class for CPPCDE values. Users can specialize this class for
// dialect-specific operations.
template <typename TOps, typename CDEValue>
class CDEValueImpl {

  template <typename TBinOp>
  CDEValue execBinOp(CDEValue other) {
    return CDEValue(this->ctx,
                    this->ctx->b.template create<TBinOp>(
                        this->get().getLoc(), this->get(), other.get()));
  }

public:
  using OpTypes = TOps;
  CDEValueImpl() {
    // Null value.
  }
  explicit CDEValueImpl(const CDEValue &other) : ctx(other.ctx) {
    this->valueOrBackedge = other.valueOrBackedge;
  }
  CDEValueImpl(CDEState *ctx, Value v) : ctx(ctx), valueOrBackedge(v) {}
  CDEValueImpl(CDEState *ctx, CDEValue v) : ctx(ctx) {
    valueOrBackedge = v.valueOrBackedge;
  }
  CDEValueImpl(CDEState *ctx, const std::shared_ptr<Backedge> &b)
      : ctx(ctx), valueOrBackedge(b) {}

  virtual ~CDEValueImpl() {}

  bool isNull() { return ctx == nullptr; }

  // Returns a registered version of this value. If no clock is provided,
  // it is assumed that the context contains a clock value.
  CDEValue reg(StringRef name, CDEValue rstValue = CDEValue(),
               CDEValue clk = CDEValue(), CDEValue rst = CDEValue()) {
    Value clock;
    if (clk.isNull()) {
      assert(ctx->clk && "A clock must be in the CDE context to creae a "
                         "register.");
      clock = ctx->clk;
    } else {
      clock = clk.get();
    }

    if (rstValue.isNull()) {
      return CDEValue(
          ctx, ctx->b.create<seq::CompRegOp>(get().getLoc(), get(), clock,
                                             ctx->b.getStringAttr(name)));
    } else {
      Value reset = rst.isNull() ? ctx->rst : rst.get();
      return CDEValue(ctx,
                      ctx->b.create<seq::CompRegOp>(
                          get().getLoc(), get().getType(), get(), clock, name,
                          reset, rstValue.get(), mlir::StringAttr()));
    }
  }

  // An explicit function for backedge assignment instead of overloading the
  // '=' operator.
  // By this, we avoid aliasing with the implicit copy constructor + don't mix
  // C++ and the underlying op generation semantics.
  void assign(CDEValue rhs) {
    auto *backedge = std::get_if<std::shared_ptr<Backedge>>(&valueOrBackedge);
    assert(backedge && "Cannot assign to a value.");
    assert(!(*backedge)->isSet() && "backedge already assigned.");
    (*backedge)->setValue(rhs.get());
  }

  // As an alternative to static_cast when implicit conversion won't apply.
  Value get() const {
    auto *value = std::get_if<Value>(&valueOrBackedge);
    if (value)
      return *value;

    return **std::get_if<std::shared_ptr<Backedge>>(&valueOrBackedge);
  }

  // Various binary operators.
  // @todo: These only really make sense for bitvector operations... The dilemma
  // is to choose between having the user forced to cast values to a CDEValue
  // specialization (e.g. CDEBitVectorValue) which then contains the operators -
  // of which use is guaranteed to be correct - or to make the API less
  // verbose by allowing all operators on all CDE values, and let type checking
  // be done at CIRCT verification time.
  // For this initial version, just use the latter.

  virtual CDEValue operator+(CDEValue other) {
    return execBinOp<typename TOps::Add>(other);
  }
  virtual CDEValue operator-(CDEValue other) {
    return execBinOp<typename TOps::Sub>(other);
  }
  virtual CDEValue operator*(CDEValue other) {
    return execBinOp<typename TOps::Mul>(other);
  }
  virtual CDEValue operator<<(CDEValue other) {
    return execBinOp<typename TOps::Shl>(other);
  }
  virtual CDEValue operator&(CDEValue other) {
    return execBinOp<typename TOps::And>(other);
  }
  virtual CDEValue operator|(CDEValue other) {
    return execBinOp<typename TOps::Or>(other);
  }
  virtual CDEValue operator^(CDEValue other) {
    return execBinOp<typename TOps::XOr>(other);
  }
  virtual CDEValue operator~() {
    return CDEValue(
        this->ctx,
        comb::createOrFoldNot(this->get().getLoc(), this->get(), this->ctx->b));
  }
  virtual CDEValue shrs(CDEValue other) {
    return execBinOp<typename TOps::ShrS>(other);
  }
  virtual CDEValue shru(CDEValue other) {
    return execBinOp<typename TOps::ShrU>(other);
  }
  virtual CDEValue divu(CDEValue other) {
    return execBinOp<typename TOps::DivU>(other);
  }
  virtual CDEValue divs(CDEValue other) {
    return execBinOp<typename TOps::DivS>(other);
  }
  virtual CDEValue modu(CDEValue other) {
    return execBinOp<typename TOps::ModU>(other);
  }
  virtual CDEValue mods(CDEValue other) {
    return execBinOp<typename TOps::ModS>(other);
  }
  virtual CDEValue concat(CDEValue other) {
    return execBinOp<typename TOps::Concat>(other);
  }

protected:
  CDEState *ctx = nullptr;
  std::variant<Value, std::shared_ptr<Backedge>> valueOrBackedge;
};

class ESICDEValue : public CDEValueImpl<CDECombOperationSelector, ESICDEValue> {
public:
  using CDEValueImpl::CDEValueImpl;

  std::pair<ESICDEValue, ESICDEValue> wrap(ESICDEValue valid) {
    auto wrapOp = ctx->b.create<esi::WrapValidReadyOp>(
        get().getLoc(), this->get(), valid.get());
    return std::make_pair(ESICDEValue(ctx, wrapOp.getResult(0)),
                          ESICDEValue(ctx, wrapOp.getResult(1)));
  }

  std::pair<ESICDEValue, ESICDEValue> unwrap(ESICDEValue ready) {
    auto unwrapOp = this->ctx->b.template create<esi::UnwrapValidReadyOp>(
        this->get().getLoc(), this->get(), ready.get());
    return std::make_pair(ESICDEValue(this->ctx, unwrapOp.getResult(0)),
                          ESICDEValue(this->ctx, unwrapOp.getResult(1)));
  }
};

struct DefaultCDEValue
    : public CDEValueImpl<CDECombOperationSelector, DefaultCDEValue> {
  using CDEValueImpl::CDEValueImpl;
};

template <typename CDEValue>
class CDEPorts {
public:
  CDEValue operator[](llvm::StringRef name) {
    auto it = ports.find(name.str());
    assert(it != ports.end() && "Port not found.");
    return it->second;
  }

protected:
  std::map<std::string, CDEValue> ports;
};

// A class for providing backedge-based access to the in- and output ports of
// a module.
template <typename CDEValue>
class CDEModulePorts : public CDEPorts<CDEValue> {

public:
  CDEModulePorts(CDEState &ctx, hw::HWModuleOp module) : ctx(ctx) {
    OpBuilder::InsertionGuard g(ctx.b);
    auto modPorts = module.getPorts();

    for (auto [barg, info] : llvm::zip(module.getArguments(), modPorts.inputs))
      this->ports.try_emplace(info.name.str(), CDEValue(&ctx, barg));

    auto outputOp = cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
    assert(outputOp.getNumOperands() == 0);

    ctx.b.setInsertionPoint(outputOp);
    llvm::SmallVector<Value> outputOpArgs;
    for (auto &info : modPorts.outputs) {
      auto be = std::make_shared<Backedge>(ctx.bb.get(info.type));
      outputBackedges.push_back(be);
      assert(this->ports.count(info.name.str()) == 0 &&
             "output port already exists");
      this->ports[info.name.str()] = CDEValue(&ctx, be);
      outputOpArgs.push_back(*be.get());
    }

    ctx.b.create<hw::OutputOp>(outputOp.getLoc(), outputOpArgs);
    outputOp.erase();
  }

private:
  CDEState &ctx;
  std::vector<std::shared_ptr<Backedge>> outputBackedges;
};

// A class facilitating CPPCDE-value access to module instance ports.
template <typename CDEValue>
class GeneratedModuleInstance : public CDEPorts<CDEValue> {

public:
  GeneratedModuleInstance(CDEState &ctx, hw::InstanceOp instance)
      : instance(instance), ctx(ctx) {
    for (size_t i = 0; i < instance.getNumResults(); ++i)
      this->ports[instance.getResultName(i).str()] =
          CDEValue(&ctx, instance.getResult(i));
  }

private:
  hw::InstanceOp instance;
  CDEState &ctx;
};

// A CDE Generator is a utility class for supporting syntactically sugar'ed
// building of CIRCT RTL dialect operations.
template <typename CDEValue>
class Generator {
public:
  using CDEModulePorts = CDEModulePorts<CDEValue>;
  Generator(Location loc, OpBuilder &b)
      : loc(loc), bb(b, loc), ctx(loc, b, bb) {}

  virtual ~Generator() = default;

protected:
  // Executes an N-ary operation on a range of CDEValues and returns the
  // result as a CDEValue.
  template <typename TOp>
  CDEValue execNAryOp(llvm::ArrayRef<CDEValue> operands) {
    return CDEValue(
        &ctx, ctx.b.create<TOp>(operands[0].get().getLoc(), unwrap(operands)));
  }

  // Various N-ary operations.
  CDEValue And(llvm::ArrayRef<CDEValue> operands) {
    return execNAryOp<typename CDEValue::OpTypes::And>(operands);
  }

  CDEValue Or(llvm::ArrayRef<CDEValue> operands) {
    return execNAryOp<typename CDEValue::OpTypes::Or>(operands);
  }

  CDEValue Xor(llvm::ArrayRef<CDEValue> operands) {
    return execNAryOp<typename CDEValue::OpTypes::XOr>(operands);
  }

  CDEValue constant(size_t width, int64_t value) {
    return CDEValue(&ctx,
                    ctx.b.create<hw::ConstantOp>(ctx.loc, APInt(width, value)));
  }

  CDEValue wire(Type t) {
    return CDEValue(&ctx, std::make_shared<Backedge>(ctx.bb.get(t)));
  }

  // Generic function for building arbitrary types through the OpBuilder.
  template <typename T, typename... Args>
  Type type(Args... args) {
    return T::get(ctx.b.getContext(), args...);
  }

  Location loc;
  BackedgeBuilder bb;
  CDEState ctx;
};

/// A CDE Generator which generates hw::HWModuleOp modules.
template <typename TGeneratorImpl, typename CDEValue>
class HWModuleGenerator : public Generator<CDEValue> {

  // Wrapper class for a CPPCDE generated module.
  class GeneratedModule {
    friend class HWModuleGenerator;

  public:
    hw::HWModuleOp get() { return module; }
    GeneratedModuleInstance<CDEValue>
    instantiate(StringRef instanceName, llvm::ArrayRef<CDEValue> operands) {
      auto unwrappedOperands = unwrap(operands);
      return GeneratedModuleInstance<CDEValue>(
          ctx, ctx.b.create<hw::InstanceOp>(ctx.b.getUnknownLoc(), module,
                                            instanceName, unwrappedOperands));
    }

  private:
    GeneratedModule(CDEState &ctx, hw::HWModuleOp module)
        : ctx(ctx), module(module) {}
    CDEState &ctx;
    hw::HWModuleOp module;
  };

public:
  HWModuleGenerator(Location loc, OpBuilder &b)
      : Generator<CDEValue>(loc, b), portInfo({}) {}

  // Run module generation.
  template <typename... Args>
  FailureOr<GeneratedModule> operator()(Args... args) {
    OpBuilder::InsertionGuard g(this->ctx.b);

    // Regardless of where the current insertion point is, the generator is
    // always expected to emit a hw::HWModuleOp in the nearest module scope.
    auto parentOp = this->ctx.b.getInsertionBlock()->getParentOp();
    if (!isa<mlir::ModuleOp>(parentOp))
      parentOp = parentOp->template getParentOfType<mlir::ModuleOp>();
    this->ctx.b.setInsertionPointToStart(
        cast<mlir::ModuleOp>(parentOp).getBody());

    auto *impl = static_cast<TGeneratorImpl *>(this);

    // Reset I/O.
    portInfo = hw::ModulePortInfo({});
    impl->createIO(args...);

    // Create the module.
    auto name = impl->getName(args...);
    auto module = this->ctx.b.template create<hw::HWModuleOp>(
        this->loc, this->ctx.b.getStringAttr(name), portInfo);

    // Generate module port accessors.
    this->ctx.b.setInsertionPointToStart(module.getBodyBlock());
    auto ports = CDEModulePorts<CDEValue>(this->ctx, module);

    if (clockIdx.has_value())
      this->ctx.clk = module.getArgument(clockIdx.value());
    if (resetIdx.has_value())
      this->ctx.rst = module.getArgument(resetIdx.value());

    // Go generate!
    this->ctx.b.setInsertionPoint(module.getBodyBlock()->getTerminator());
    impl->generate(ports, args...);
    return GeneratedModule(this->ctx, module);
  }

protected:
  // Adds an input port to this module.
  void input(llvm::StringRef name, Type type,
             PortKind kind = PortKind::Default) {
    size_t idx = portInfo.inputs.size();
    portInfo.inputs.push_back(hw::PortInfo{
        this->ctx.b.getStringAttr(name), hw::PortDirection::INPUT, type, idx});
    if (kind == PortKind::Clock) {
      assert(!clockIdx.has_value() && "multiple clocks not allowed");
      clockIdx = idx;
    } else if (kind == PortKind::Reset) {
      assert(!resetIdx.has_value() && "multiple resets not allowed");
      resetIdx = idx;
    }
  }

  // Shorthand for creating a clock input port.
  void clock(llvm::StringRef name = "clk") {
    input(name, this->ctx.b.getIntegerType(1), PortKind::Clock);
  }

  // Shorthand for creating a reset input port.
  void reset(llvm::StringRef name = "rst") {
    input(name, this->ctx.b.getIntegerType(1), PortKind::Reset);
  }

  // Adds an output port to this module.
  void output(llvm::StringRef name, Type type) {
    portInfo.outputs.push_back(hw::PortInfo{this->ctx.b.getStringAttr(name),
                                            hw::PortDirection::OUTPUT, type,
                                            portInfo.outputs.size()});
  }

  hw::ModulePortInfo portInfo;

private:
  // Indices of the clock and reset ports in the generated module.
  std::optional<size_t> clockIdx, resetIdx;
};

} // namespace cppcde
} // namespace circt
