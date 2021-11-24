#ifndef CIRCT_TOOLS_HLT_HANDSHAKESIMINTERFACE_H
#define CIRCT_TOOLS_HLT_HANDSHAKESIMINTERFACE_H

#include "circt/Tools/hlt/Simulator/VerilatorSimInterface.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace circt {
namespace hlt {

struct TransactableTrait {
  enum State {
    // No transaction
    Idle,
    // Transaction is in progress (expecting Transacted to be set during eval())
    TransactNext,
    // Just transacted.
    Transacted
  };
  // If this is set, the port was transacted in the last cycle.
  State txState = Idle;
  bool transacted() { return txState == State::Transacted; }
};

template <typename TSimPort>
struct HandshakePort : public TSimPort, public TransactableTrait {
  HandshakePort() {}
  HandshakePort(CData *readySig, CData *validSig)
      : readySig(readySig), validSig(validSig){};
  HandshakePort(const std::string &name, CData *readySig, CData *validSig)
      : name(name), readySig(readySig), validSig(validSig){};

  void dump(std::ostream &out) const {
    out << (name.empty() ? "?" : name) << ">\t";
    out << "r: " << static_cast<int>(*readySig)
        << "\tv: " << static_cast<int>(*validSig);
  }
  bool valid() { return *(this->validSig) == 1; }
  bool ready() { return *(this->readySig) == 1; }

  CData *readySig = nullptr;
  CData *validSig = nullptr;
  std::string name;
};

struct HandshakeInPort : public HandshakePort<SimulatorInPort> {
  using HandshakePort<SimulatorInPort>::HandshakePort;
  void reset() override { *(this->validSig) = !1; }
  bool ready() override {
    // An input port is ready to accept inputs when an input is not already
    // pushed onto the port (validSig == 1).
    return *(this->validSig) == 0;
  }

  // Writing to an input port implies setting the valid signal.
  virtual void write() { *(this->validSig) = !0; }

  /// An input port transaction is fulfilled by de-asserting the valid (output)
  // signal of his handshake bundle.
  void eval() override {
    if (txState == Transacted)
      txState = Idle;

    bool transacting = txState == TransactNext;
    if (transacting || *(this->validSig) && *(this->readySig)) {
      if (transacting) {
        *this->validSig = 0;
        txState = Transacted;
      } else
        txState = TransactNext;
    }
  }
};

struct HandshakeOutPort : public HandshakePort<SimulatorOutPort> {
  using HandshakePort<SimulatorOutPort>::HandshakePort;
  void reset() override { *(this->readySig) = !1; }
  virtual void read() {
    // todo
  }

  /// An output port transaction is fulfilled whenever the ready signal of the
  // handshake bundle is asserted and the valid signal is not. A precondition
  // is that the valid signal was asserter before the ready signal.
  void eval() override {
    if (txState == Transacted)
      txState = Idle;

    bool transacting = txState == TransactNext;
    if (transacting || *(this->validSig) && *(this->readySig)) {
      if (transacting) {
        // *this->readySig = 0;
        txState = Transacted;
      } else
        txState = TransactNext;
    }
  }
};

template <typename TData, typename THandshakeIOPort>
struct HandshakeDataPort : public THandshakeIOPort {
  HandshakeDataPort() {}
  HandshakeDataPort(CData *readySig, CData *validSig, TData *dataSig)
      : THandshakeIOPort(readySig, validSig), dataSig(dataSig){};
  HandshakeDataPort(const std::string &name, CData *readySig, CData *validSig,
                    TData *dataSig)
      : THandshakeIOPort(name, readySig, validSig), dataSig(dataSig){};
  void dump(std::ostream &out) const {
    THandshakeIOPort::dump(out);
    out << "\t" << static_cast<int>(*dataSig);
  }
  TData *dataSig = nullptr;
};

template <typename TData>
struct HandshakeDataInPort : HandshakeDataPort<TData, HandshakeInPort> {
  using HandshakeDataPortImpl = HandshakeDataPort<TData, HandshakeInPort>;
  using HandshakeDataPortImpl::HandshakeDataPortImpl;
  void writeData(TData in) {
    HandshakeDataPortImpl::write();
    *(this->dataSig) = in;
  }
};

template <typename TData>
struct HandshakeDataOutPort : HandshakeDataPort<TData, HandshakeOutPort> {
  using HandshakeDataPortImpl = HandshakeDataPort<TData, HandshakeOutPort>;
  using HandshakeDataPortImpl::HandshakeDataPortImpl;
  TData readData() {
    HandshakeDataPortImpl::read();
    return *(this->dataSig);
  }
};

// A HandshakeMemoryInterface represents a wrapper around a
// handshake.extmemory operation. It is initialized with a set of load- and
// store ports which, when transacting, will access the pointer provided to
// the memory interface during simulation. The memory interface inherits from
// SimulatorInPort due to handshake circuits receiving a memory interface as a
// memref input.
template <typename TData, typename TAddr>
class HandshakeMemoryInterface : public SimulatorInPort,
                                 public TransactableTrait {
  struct StorePort {
    std::shared_ptr<HandshakeDataOutPort<TData>> data;
    std::shared_ptr<HandshakeDataOutPort<TAddr>> addr;
    std::shared_ptr<HandshakeInPort> done;
    void eval() {
      data->eval();
      addr->eval();
      done->eval();
    }
  };

  struct LoadPort {
    std::shared_ptr<HandshakeDataInPort<TData>> data;
    std::shared_ptr<HandshakeDataOutPort<TAddr>> addr;
    std::shared_ptr<HandshakeInPort> done;
    void eval() {
      data->eval();
      addr->eval();
      done->eval();
    }
  };

public:
  // A memory interface is initialized with a static memory size. This is
  // generated during wrapper generation.
  HandshakeMemoryInterface(size_t size) : memorySize(size) {}

  void dump(std::ostream &os) const {}

  void setMemory(void *memory) {
    if (memory_ptr != nullptr)
      assert(memory_ptr == memory &&
             "The memory should always point to the same base address "
             "throughout simulation.");
    memory_ptr = reinterpret_cast<TData *>(memory);
    txState = TransactNext;
  }

  virtual ~HandshakeMemoryInterface() = default;

  void
  addStorePort(const std::shared_ptr<HandshakeDataOutPort<TData>> &dataPort,
               const std::shared_ptr<HandshakeDataOutPort<TAddr>> &addrPort,
               const std::shared_ptr<HandshakeInPort> &donePort) {
    storePorts.push_back(StorePort{dataPort, addrPort, donePort});
  }

  void addLoadPort(const std::shared_ptr<HandshakeDataInPort<TData>> &dataPort,
                   const std::shared_ptr<HandshakeDataOutPort<TAddr>> &addrPort,
                   const std::shared_ptr<HandshakeInPort> &donePort) {
    loadPorts.push_back(LoadPort{dataPort, addrPort, donePort});
  }

  void reset() override {
    for (auto &port : storePorts) {
      *(port.data->validSig) = !1;
      *(port.addr->validSig) = !1;
      *(port.done->readySig) = !1;
    }
    for (auto &port : loadPorts) {
      *(port.data->readySig) = !1;
      *(port.addr->validSig) = !1;
      *(port.done->readySig) = !1;
    }
  }
  bool ready() override {
    assert(false && "N/A for memory interfaces.");
    return false;
  }

  // Writing to an input port implies setting the valid signal.
  virtual void write() { assert(false && "N/A for memory interfaces."); }

  /// An input port transaction is fulfilled by de-asserting the valid
  /// (output)
  // signal of his handshake bundle.
  void eval() override {
    switch (txState) {
    case Idle:
      break;
    case TransactNext:
      txState = Transacted;
      break;
    case Transacted:
      txState = Idle;
      break;
    }

    // Current cycle transactions:
    // Load ports
    for (auto &loadPort : loadPorts) {
      if (*(loadPort.addr->validSig) && *(loadPort.data->readySig) &&
          *(loadPort.done->readySig)) {
        assert(memory_ptr != nullptr && "Memory not set.");
        *(loadPort.addr->readySig) = 1;
        *(loadPort.data->validSig) = 1;
        *(loadPort.done->validSig) = 1;
        size_t addr = *(loadPort.addr->dataSig);
        assert(addr < memorySize && "Address out of bounds.");
        *(loadPort.data->dataSig) = memory_ptr[addr];
      }
    }

    // Store ports
    for (auto &storePort : storePorts) {
      if (*(storePort.addr->validSig) && *(storePort.data->validSig) &&
          *(storePort.done->readySig)) {
        assert(memory_ptr != nullptr && "Memory not set.");
        *(storePort.addr->readySig) = 1;
        *(storePort.data->readySig) = 1;
        *(storePort.done->validSig) = 1;
        size_t addr = *(storePort.addr->dataSig);
        assert(addr < memorySize && "Address out of bounds.");
        memory_ptr[addr] = *(storePort.data->dataSig);
      }
    }

    for (auto &port : storePorts)
      port.eval();
    for (auto &port : loadPorts)
      port.eval();
  }

private:
  TData readMemory(uint32_t addr) {
    assert(memory_ptr != nullptr && "Memory not set.");
    assert(addr < memorySize && "Address out of bounds.");
    return memory_ptr[addr];
  }
  void writeMemory(uint32_t addr, TData data) {
    assert(memory_ptr != nullptr && "Memory not set.");
    assert(addr < memorySize && "Address out of bounds.");
    memory_ptr[addr] = data;
  }

  std::vector<StorePort> storePorts;
  std::vector<LoadPort> loadPorts;

  // The memory pointer is set by the simulation engine during execution.
  TData *memory_ptr = nullptr;

  // The size of the memory associated with this interface.
  size_t memorySize;
};

template <typename TInput, typename TOutput, typename TModel>
class HandshakeSimInterface
    : public VerilatorSimInterface<TInput, TOutput, TModel> {
public:
  using VerilatorSimImpl = VerilatorSimInterface<TInput, TOutput, TModel>;

  template <typename T>
  struct TransactionBuffer {
    TransactionBuffer(const T &data = T()) : data(data) {
      for (int i = 0; i < std::tuple_size<T>(); ++i)
        transacted[i] = false;
    }
    T data;
    // Maintain a mapping between the index of each subtype in data and
    // whether that subtype has been transacted.
    std::map<unsigned, bool> transacted;
    // Flag to indicate if the input control has been transacted for this
    // buffer.
    bool transactedControl = false;
  };

  struct InputBuffer : public TransactionBuffer<TInput> {
    InputBuffer(const TInput &data) : TransactionBuffer<TInput>(data) {}

    bool done() {
      return this->transactedControl &&
             std::all_of(this->transacted.begin(), this->transacted.end(),
                         [](const auto &pair) { return pair.second; });
    }
  };

  struct OutputBuffer : public TransactionBuffer<TOutput> {
    OutputBuffer() : TransactionBuffer<TOutput>() {}

    bool valid() {
      return this->transactedControl &&
             std::all_of(this->transacted.begin(), this->transacted.end(),
                         [](const auto &pair) { return pair.second; });
    }
  };

  HandshakeSimInterface() : VerilatorSimImpl() {
    // Allocate in- and output control ports.
    inCtrl = std::make_unique<HandshakeInPort>();
    outCtrl = std::make_unique<HandshakeOutPort>();
  }

  // The handshake simulator is ready to accept inputs whenever it is not
  // currently transacting an input buffer.
  bool inReady() override { return !this->inBuffer.has_value(); }

  // The handshake simulator is ready to provide an output whenever it has
  // a valid output buffer.
  bool outValid() override { return this->outBuffer.valid(); }

  void step() override {
    // Rising edge
    VerilatorSimImpl::clock_rising();

    readToOutputBuffer();
    writeFromInputBuffer();

    // Advance time for debugging purposes; separates the clk edge from changes
    // to the following signals.
    this->advanceTime();
    // Transact all I/O ports
    for (auto &port : llvm::enumerate(this->outPorts)) {
      port.value()->eval();
      auto transactable = dynamic_cast<TransactableTrait *>(port.value().get());
      assert(transactable);
      if (transactable->transacted())
        outBuffer.transacted[port.index()] = true;
    }
    for (auto &port : llvm::enumerate(this->inPorts)) {
      port.value()->eval();
      auto transactable = dynamic_cast<TransactableTrait *>(port.value().get());
      assert(transactable);
      if (transactable->transacted())
        inBuffer.value().transacted[port.index()] = true;
    }

    // Transact control ports
    inCtrl->eval();
    if (inCtrl->transacted())
      inBuffer.value().transactedControl = true;

    outCtrl->eval();
    if (outCtrl->transacted())
      outBuffer.transactedControl = true;

    // Set output ports readyness based on which outputs in the current output
    // buffer have been transacted.
    for (auto outPort : llvm::enumerate(this->outPorts)) {
      auto outPortp = dynamic_cast<HandshakeOutPort *>(outPort.value().get());
      assert(outPortp);
      if (!outBuffer.transacted[outPort.index()])
        *(outPortp->readySig) = 1;
    }
    if (!outBuffer.transactedControl)
      *(outCtrl->readySig) = 1;

    // Falling edge
    VerilatorSimImpl::clock_falling();
    this->advanceTime();
  }

  void setup() override {
    inCtrl->name = "inCtrl";
    outCtrl->name = "outCtrl";
    assert(inCtrl->readySig != nullptr && "Missing in control ready signal");
    assert(inCtrl->validSig != nullptr && "Missing in control valid signal");
    assert(outCtrl->readySig != nullptr && "Missing out control ready signal");
    assert(outCtrl->validSig != nullptr && "Missing out control valid signal");

    // Do verilator initialization; this will reset the circuit
    VerilatorSimImpl::setup();

    // Run a few cycles to ensure everything works after the model is out of
    // reset and a subset of all ports are ready/valid.
    for (int i = 0; i < 2; ++i)
      VerilatorSimImpl::clock();
  }

  void dump(std::ostream &out) const override {
    out << "Control states:\n";
    out << *inCtrl << "\n";
    out << *outCtrl << "\n";
    VerilatorSimImpl::dump(out);
  }

  template <std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  writeInputRec(const std::tuple<Tp...> &) {
    // End-case, do nothing
  }

  template <std::size_t I = 0, typename... Tp>
      inline
      typename std::enable_if < I<sizeof...(Tp), void>::type
                                writeInputRec(const std::tuple<Tp...> &tInput) {
    auto value = std::get<I>(tInput);
    auto &inBufferV = inBuffer.value();

    // Is this a simple input port?
    auto p = this->inPorts.at(I).get();
    if (!inBufferV.transacted[I]) {
      // Normal port?
      if (auto inPort = dynamic_cast<HandshakeDataInPort<decltype(value)> *>(p);
          inPort) {
        // A value can be written to an input port when it is not already trying
        // to transact a value.
        if (!inPort->valid()) {
          inPort->writeData(value);
        }
      }
      // Memory interface?
      else if (auto inMemPort = dynamic_cast<HandshakeMemoryInterface<
                   std::remove_pointer_t<decltype(value)>, QData> *>(p);
               inMemPort) {
        inMemPort->setMemory(reinterpret_cast<void *>(value));
      } else {
        assert(false && "Unsupported input port type");
      }
    }

    writeInputRec<I + 1, Tp...>(tInput);
  }

  void writeFromInputBuffer() {
    if (!inBuffer.has_value())
      return; // Nothing to transact.

    auto &inBufferV = inBuffer.value();

    // Try writing input data.
    writeInputRec(inBufferV.data);

    // Try writing input control.
    if (!inBufferV.transactedControl)
      inCtrl->write();

    // Finish writing input buffer?
    if (inBufferV.done())
      inBuffer = std::nullopt;
  }

  void pushInput(const TInput &v) override {
    assert(!inBuffer.has_value() &&
           "pushing input while already having an input buffer?");
    inBuffer = {v};
  }

  template <std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  readOutputRec(std::tuple<Tp...> &) {
    // End-case, do nothing
  }

  template <std::size_t I = 0, typename... Tp>
      inline typename std::enable_if <
      I<sizeof...(Tp), void>::type readOutputRec(std::tuple<Tp...> &tOutput) {
    using ValueType = std::remove_reference_t<decltype(std::get<I>(tOutput))>;
    auto outPort = dynamic_cast<HandshakeDataOutPort<ValueType> *>(
        this->outPorts.at(I).get());
    assert(outPort);
    if (outPort->valid() && outPort->ready()) {
      std::get<I>(tOutput) = outPort->readData();
    }
    readOutputRec<I + 1, Tp...>(tOutput);
  }

  void readToOutputBuffer() {
    if (outBuffer.valid())
      return; // Nothing to transact.

    // Try reading output data.
    readOutputRec(outBuffer.data);

    // OutBuffer will be cleared by popOutput if all data has been read.
  }

  TOutput popOutput() override {
    assert(outBuffer.valid() && "popping output buffer that is not valid?");
    auto vOutput = outBuffer.data;
    outBuffer = OutputBuffer(); // reset
    return vOutput;
  }

protected:
  // Handshake interface signals. Defined as raw pointers since they are owned
  // by VerilatorSimInterface.
  std::unique_ptr<HandshakeInPort> inCtrl;
  std::unique_ptr<HandshakeOutPort> outCtrl;

  // In- and output buffers.
  // @todo: this could be made into separate buffers for each subtype within
  // TInput and TOutput, allowing for decoupling of starting the writing of a
  // new input buffer until all values within an input have been transacted.
  std::optional<InputBuffer> inBuffer;
  OutputBuffer outBuffer;
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_HANDSHAKESIMINTERFACE_H
