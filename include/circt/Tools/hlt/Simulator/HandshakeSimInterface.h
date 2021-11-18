#ifndef CIRCT_TOOLS_HLT_HANDSHAKESIMINTERFACE_H
#define CIRCT_TOOLS_HLT_HANDSHAKESIMINTERFACE_H

#include "circt/Tools/hlt/Simulator/VerilatorSimInterface.h"

namespace circt {
namespace hlt {

template <typename TSimPort>
struct HandshakePort : public TSimPort {
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

  virtual void transact() = 0;

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
  void transact() override {
    if (*(this->validSig) && *(this->readySig))
      *this->validSig = !1;
  }
};

struct HandshakeOutPort : public HandshakePort<SimulatorOutPort> {
  using HandshakePort<SimulatorOutPort>::HandshakePort;
  void reset() override { *(this->readySig) = !1; }
  bool valid() override { return *(this->validSig) == 1; }
  virtual void read() {
    // todo
  }

  /// An input port transaction is fulfilled by de-asserting the valid (output)
  // signal of his handshake bundle.
  void transact() override {
    // Edit: by making the ready signal always enabled, we avoid stalling the
    // handshake model.
    // if (*(this->validSig) && *(this->readySig))
    //   *this->readySig = !1;
  }
};

template <typename TData, typename THandshakeIOPort>
struct HandshakeDataPort : public THandshakeIOPort {
  static_assert(!std::is_pointer<TData>::value &&
                    !std::is_reference<TData>::value,
                "Must not be a pointer or reference type");
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

// A HandshakeMemoryInterface represents a wrapper around a handshake.extmemory
// operation. It is initialized with a set of load- and store ports which, when
// transacting, will access the pointer provided to the memory interface during
// simulation. The memory interface inherits from SimulatorInPort due to
// handshake circuits receiving a memory interface as a memref input.
template <typename TData>
class HandshakeMemoryInterface : SimulatorInPort {

  struct StorePort {
    std::shared_ptr<HandshakeDataInPort<TData>> data;
    std::shared_ptr<HandshakeInPort> addr;
    std::shared_ptr<HandshakeOutPort> done;
  };

  struct LoadPort {
    std::shared_ptr<HandshakeInPort> addr;
    std::shared_ptr<HandshakeOutPort> done;
    std::shared_ptr<HandshakeDataOutPort<TData>> data;
  };

public:
  HandshakeMemoryInterface(size_t size) : memorySize(size) {}

  void setMemory(TData *memory) {
    if (memory_ptr != nullptr)
      assert(memory_ptr == memory &&
             "The memory should always point to the same base address "
             "throughout simulation.");
    memory_ptr = memory;
  }

  void transact() {}

  virtual ~HandshakeMemoryInterface() = default;

  void addStorePort(std::shared_ptr<HandshakeInPort> &dataPort,
                    std::shared_ptr<HandshakeInPort> &addrPort,
                    std::shared_ptr<HandshakeOutPort> &donePort) {
    storePorts.push_back(StorePort{dataPort, addrPort, donePort});
  }

  void addLoadPort(std::shared_ptr<HandshakeInPort> &addrPort,
                   std::shared_ptr<HandshakeOutPort> &donePort,
                   std::shared_ptr<HandshakeOutPort> &dataPort) {
    loadPorts.push_back(LoadPort{addrPort, donePort, dataPort});
  }

private:
  TData read(uint32_t addr) {
    assert(memory_ptr != nullptr && "Memory not set.");
    assert(addr < memorySize && "Address out of bounds.");
    return memory_ptr[addr];
  }
  void write(uint32_t addr, TData data) {
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

  HandshakeSimInterface() : VerilatorSimImpl() {
    // Allocate in- and output control ports.
    inCtrl = std::make_unique<HandshakeInPort>();
    outCtrl = std::make_unique<HandshakeOutPort>();
  }

  void step() override {
    for (auto &port : this->outPorts)
      static_cast<HandshakeOutPort *>(port.get())->transact();
    VerilatorSimImpl::step();

    // Always reset the input control valid signal after clocking, to ensure
    // that only as many rounds are initialized as the number of times that
    // we've pushed inputs.
    *inCtrl->validSig = 0;

    // Transact all I/O ports
    for (auto &port : this->inPorts)
      static_cast<HandshakeInPort *>(port.get())->transact();
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

    // Set output ports as ready; this means that when input ports are valid,
    // the model will start execution.
    for (auto &outPort : this->outPorts) {
      auto outPortp = dynamic_cast<HandshakeOutPort *>(outPort.get());
      assert(outPortp);
      *(outPortp->readySig) = !0;
    }
    *(outCtrl->readySig) = !0;
    *inCtrl->validSig = !0;

    // Run a few cycles to ensure everything works after the model is out of
    // reset and a subset of all ports are ready/valid.
    for (int i = 0; i < 2; ++i)
      clock();
  }

  void dump(std::ostream &out) const override {
    out << "Control states:\n";
    out << *inCtrl << "\n";
    out << *outCtrl << "\n";
    VerilatorSimImpl::dump(out);
  }

  template <std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  pushInputRec(const std::tuple<Tp...> &) {
    // End-case, do nothing
  }

  template <std::size_t I = 0, typename... Tp>
      inline
      typename std::enable_if < I<sizeof...(Tp), void>::type
                                pushInputRec(const std::tuple<Tp...> &tInput) {
    auto value = std::get<I>(tInput);
    auto inPort = dynamic_cast<HandshakeDataInPort<decltype(value)> *>(
        this->inPorts.at(I).get());
    assert(inPort);
    inPort->writeData(value);
    pushInputRec<I + 1, Tp...>(tInput);
  }

  void pushInput(const TInput &v) override {
    pushInputRec(v);

    // Pushing inputs implies starting a new round of the kernel.
    *inCtrl->validSig = 1;
  }

  template <std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  popOutputRec(std::tuple<Tp...> &) {
    // End-case, do nothing
  }

  template <std::size_t I = 0, typename... Tp>
      inline typename std::enable_if <
      I<sizeof...(Tp), void>::type popOutputRec(std::tuple<Tp...> &tOutput) {
    using ValueType = std::remove_reference_t<decltype(std::get<I>(tOutput))>;
    auto outPort = dynamic_cast<HandshakeDataOutPort<ValueType> *>(
        this->outPorts.at(I).get());
    assert(outPort);
    std::get<I>(tOutput) = outPort->readData();
    popOutputRec<I + 1, Tp...>(tOutput);
  }

  TOutput popOutput() override {
    TOutput out;
    popOutputRec(out);
    return out;
  }

protected:
  // Handshake interface signals. Defined as raw pointers since they are owned
  // by VerilatorSimInterface.
  std::unique_ptr<HandshakeInPort> inCtrl;
  std::unique_ptr<HandshakeOutPort> outCtrl;
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_HANDSHAKESIMINTERFACE_H
