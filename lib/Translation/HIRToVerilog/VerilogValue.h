#ifndef __HIRVERILOGVALUE__
#define __HIRVERILOGVALUE__

#include "circt/Dialect/HIR/HIR.h"
#include "circt/Translation/HIRToVerilog.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <string>

using namespace mlir;
using namespace hir;
using namespace std;

static bool isIntegerType(Type type) {
  assert(type);
  if (type.isa<IntegerType>())
    return true;
  else if (type.isa<hir::ConstType>())
    return true;
  else if (auto wireType = type.dyn_cast<WireType>())
    return isIntegerType(wireType.getElementType());
  return false;
}

class VerilogValue {
private:
  SmallVector<unsigned, 4> distDims;
  SmallVector<unsigned, 4> distDimPos;
  struct {
    SmallVector<int, 1> numReads;
    SmallVector<int, 1> numWrites;
  } usageInfo;
  Type type;
  string name;
  Value value;

public:
  VerilogValue() : value(Value()), type(Type()), name("uninitialized") {
    maxDelay = 0;
  }
  VerilogValue(Value value, string name)
      : initialized(true), value(value), type(value.getType()), name(name) {
    string out;
    auto size = 1;
    if (type.isa<MemrefType>()) {
      auto packing = type.dyn_cast<hir::MemrefType>().getPacking();
      auto shape = type.dyn_cast<hir::MemrefType>().getShape();
      for (int i = 0; i < shape.size(); i++) {
        bool isDistDim = true;
        for (auto p : packing) {
          if (i == shape.size() - p - 1)
            isDistDim = false;
        }
        if (isDistDim) {
          distDims.push_back(shape[i]);
          distDimPos.push_back(i);
          size *= shape[i];
        }
      }
    }
    usageInfo.numReads.insert(usageInfo.numReads.begin(), size, 0);
    usageInfo.numWrites.insert(usageInfo.numWrites.begin(), size, 0);
  }
  ~VerilogValue() { initialized = false; }
  void reset() { initialized = false; }
  bool isInitialized() { return initialized; }

public:
  int numReads(ArrayRef<VerilogValue *> addr) {
    unsigned pos = posMemrefDistDimAccess(addr);
    return usageInfo.numReads[pos];
  }
  int numWrites(ArrayRef<VerilogValue *> addr) {
    unsigned pos = posMemrefDistDimAccess(addr);
    return usageInfo.numWrites[pos];
  }
  unsigned numAccess(unsigned pos) {
    return usageInfo.numReads[pos] + usageInfo.numWrites[pos];
  }
  unsigned numAccess(ArrayRef<VerilogValue *> addr) {
    unsigned pos = posMemrefDistDimAccess(addr);
    return numAccess(pos);
  }

public:
  unsigned maxNumReads() {
    unsigned out = 0;
    for (int i = 0; i < usageInfo.numReads.size(); i++) {
      unsigned nReads = usageInfo.numReads[i];
      // TODO: We don't yet support different number of reads to different
      // wires in array.
      if (i != 0 && out != nReads) {
        emitWarning(this->value.getLoc(),
                    "Number of reads is not same for all elements.");
      }
      out = (out > nReads) ? out : nReads;
    }
    return out;
  }
  unsigned maxNumWrites() {
    unsigned out = 0;
    for (int i = 0; i < usageInfo.numReads.size(); i++) {
      unsigned nWrites = usageInfo.numWrites[i];
      // TODO: We don't yet support different number of writes to different
      // wires in array. See printCallOp in HIRToVerilog. It increments numReads
      // for all reads.
      if (i != 0 && out != nWrites) {
        emitWarning(this->value.getLoc(),
                    "Number of writes is not same for all elements.");
      }
      out = (out > nWrites) ? out : nWrites;
    }
    return out;
  }
  unsigned maxNumAccess() {
    unsigned out = 0;
    for (int i = 0; i < usageInfo.numReads.size(); i++) {
      unsigned nAccess = usageInfo.numReads[i] + usageInfo.numWrites[i];
      // TODO: We don't yet support different number of access to different
      // wires in array.
      if (i != 0 && out != nAccess) {
        emitWarning(this->value.getLoc(),
                    "Number of access is not same for all elements.");
      }
      out = (out > nAccess) ? out : nAccess;
    }
    return out;
  }

  string strMemrefArgDecl();
  string strMemrefInstDecl() const;
  Type getType() { return type; }
  ArrayRef<unsigned> getShape() const {
    assert(type.isa<MemrefType>());
    return type.dyn_cast<MemrefType>().getShape();
  }
  ArrayRef<unsigned> getPacking() const {
    assert(type.isa<MemrefType>());
    return type.dyn_cast<MemrefType>().getPacking();
  }

  Type getElementType() const {
    assert(type.isa<MemrefType>());
    return type.dyn_cast<MemrefType>().getElementType();
  }

private:
  bool initialized = false;
  SmallVector<unsigned, 4> getMemrefPackedDims() const;

  string strMemrefDistDims() const;

  string strWireInput() { return name + "_input"; }
  string strMemrefAddrValid() const { return name + "_addr_valid"; }
  string strMemrefAddrInput() { return name + "_addr_input"; }
  string strMemrefWrDataValid() { return name + "_wr_data_valid"; }
  string strMemrefWrDataInput() { return name + "_wr_data_input"; }
  string strMemrefRdEnInput() { return name + "_rd_en_input"; }
  string strMemrefWrEnInput() { return name + "_wr_en_input"; }
  string buildEnableSelectorStr(string str_en, string str_en_input,
                                unsigned numInputs);

  string strDistDimLoopNest(string distDimIdxStub, string body);

  string strMemrefSelect(string str_v, string str_v_array, unsigned idx,
                         string str_dataWidth);

  string buildDataSelectorStr(string str_v, string str_v_valid,
                              string str_v_input, unsigned numInputs,
                              unsigned dataWidth);

  unsigned posMemrefDistDimAccess(ArrayRef<VerilogValue *> addr) const;

  string strMemrefDistDimAccess(ArrayRef<VerilogValue *> addr) const;

public:
  unsigned getMemrefPackedSize() const;

  SmallVector<unsigned, 4> getMemrefDistDims() const;
  void setIntegerConst(int value);
  bool isIntegerConst() const;
  int getIntegerConst() const;

  string strMemrefSelDecl();

  string strWire() const {
    if (name == "") {
      return "/*ERROR: Anonymus variable.*/";
    }
    return name;
  }

  unsigned getBitWidth() const;

  bool isIntegerType() const;
  bool isFloatType() const;
  /// Checks if the type is implemented as a verilog wire or an array of wires.
  bool isSimpleType() const {
    if (isIntegerType() || isFloatType())
      return true;
    else if (type.isa<hir::TimeType>())
      return true;
    return false;
  }

  string strWireDecl() const;
  string strConstOrError(int n = 0) const;

  string strConstOrWire(unsigned bitwidth) const;
  string strConstOrWire() const;

  string strWireValid() { return name + "_valid"; }
  string strWireInput(unsigned idx) {
    return strWireInput() + "[" + to_string(idx) + "]";
  }
  string strDelayedWire() const { return name + "delay"; }
  string strDelayedWire(unsigned delay) {
    updateMaxDelay(delay);
    return strDelayedWire() + "[" + to_string(delay) + "]";
  }

  string strMemrefAddr() const { return name + "_addr"; }
  string strMemrefAddrValidIf(unsigned idx) {
    return strMemrefAddrValid() + "_if[" + to_string(idx) + "]";
  }
  string strMemrefAddrValid(unsigned idx) {
    return strMemrefAddrValid() + "[" + to_string(idx) + "]";
  }
  string strMemrefAddrValid(ArrayRef<VerilogValue *> addr, unsigned idx) const {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefAddrValid() + distDimAccessStr + "[" + to_string(idx) + "]";
  }
  string strMemrefAddrInputIf(unsigned idx) {
    return strMemrefAddrInput() + "_if[" + to_string(idx) + "]";
  }
  string strMemrefAddrInput(unsigned idx) {
    return strMemrefAddrInput() + "[" + to_string(idx) + "]";
  }
  string strMemrefAddrInput(ArrayRef<VerilogValue *> addr, unsigned idx) {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefAddrInput() + distDimAccessStr + "[" + to_string(idx) + "]";
  }

  string strMemrefRdData() const { return name + "_rd_data"; }
  string strMemrefRdData(ArrayRef<VerilogValue *> addr) const {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefRdData() + distDimAccessStr;
  }

  string strMemrefWrData() const { return name + "_wr_data"; }
  string strMemrefWrDataValidIf(unsigned idx) {
    return strMemrefWrDataValid() + "_if[" + to_string(idx) + "]";
  }
  string strMemrefWrDataValid(ArrayRef<VerilogValue *> addr, unsigned idx) {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefWrDataValid() + distDimAccessStr + "[" + to_string(idx) +
           "]";
  }
  string strMemrefWrDataInputIf(unsigned idx) {
    return strMemrefWrDataInput() + "_if[" + to_string(idx) + "]";
  }
  string strMemrefWrDataInput(unsigned idx) {
    return strMemrefWrDataInput() + "[" + to_string(idx) + "]";
  }
  string strMemrefWrDataInput(ArrayRef<VerilogValue *> addr, unsigned idx) {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefWrDataInput() + distDimAccessStr + "[" + to_string(idx) +
           "]";
  }

  string strMemrefRdEn() const { return name + "_rd_en"; }
  string strMemrefRdEnInputIf(unsigned idx) {
    return strMemrefRdEnInput() + "_if[" + to_string(idx) + "]";
  }
  string strMemrefRdEnInput(unsigned idx) {
    return strMemrefRdEnInput() + "[" + to_string(idx) + "]";
  }
  string strMemrefRdEnInput(ArrayRef<VerilogValue *> addr, unsigned idx) {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefRdEnInput() + distDimAccessStr + "[" + to_string(idx) + "]";
  }

  string strMemrefWrEn() const { return name + "_wr_en"; }
  string strMemrefWrEnInputIf(unsigned idx) {
    return strMemrefWrEnInput() + "_if[" + to_string(idx) + "]";
  }
  string strMemrefWrEnInput(unsigned idx) {
    return strMemrefWrEnInput() + "[" + to_string(idx) + "]";
  }
  string strMemrefWrEnInput(ArrayRef<VerilogValue *> addr, unsigned idx) {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefWrEnInput() + distDimAccessStr + "[" + to_string(idx) + "]";
  }
  void incMemrefNumReads() {
    for (int pos = 0; pos < usageInfo.numReads.size(); pos++) {
      usageInfo.numReads[pos]++;
    }
  }
  void incMemrefNumWrites() {
    for (int pos = 0; pos < usageInfo.numWrites.size(); pos++) {
      usageInfo.numWrites[pos]++;
    }
  }
  void incMemrefNumReads(ArrayRef<VerilogValue *> addr) {
    unsigned pos = posMemrefDistDimAccess(addr);
    usageInfo.numReads[pos]++;
  }
  void incMemrefNumWrites(ArrayRef<VerilogValue *> addr) {
    unsigned pos = posMemrefDistDimAccess(addr);
    usageInfo.numWrites[pos]++;
  }

  int getMaxDelay() const {
    if (maxDelay < 0 || maxDelay > 64) {
      fprintf(stderr, "unexpected maxDelay %d\n", maxDelay);
    }
    return maxDelay;
  }

private:
  void updateMaxDelay(int delay) {
    if (delay < 0 || delay > 64) {
      fprintf(stderr, "unexpected delay %d\n", delay);
    }
    maxDelay = (maxDelay > delay) ? maxDelay : delay;
  }
  bool isConstValue = false;
  union {
    int val_int;
  } constValue;

  int maxDelay = 0;
};

string VerilogValue::strMemrefInstDecl() const {
  string out_decls;
  MemrefType memrefTy = type.dyn_cast<MemrefType>();
  hir::Details::PortKind port = memrefTy.getPort();
  string portString =
      ((port == hir::Details::r) ? "r"
                                 : (port == hir::Details::w) ? "w" : "rw");
  unsigned addrWidth = helper::calcAddrWidth(memrefTy);
  string distDimsStr = strMemrefDistDims();
  unsigned dataWidth = helper::getBitWidth(memrefTy.getElementType());
  if (addrWidth > 0) { // All dims may be distributed.
    out_decls += "reg[" + to_string(addrWidth - 1) + ":0] " + strMemrefAddr() +
                 distDimsStr + ";\n";
  }
  if (port == hir::Details::r || port == hir::Details::rw) {
    out_decls += "wire " + strMemrefRdEn() + distDimsStr + ";\n";
    out_decls += "logic[" + to_string(dataWidth - 1) + ":0] " +
                 strMemrefRdData(SmallVector<VerilogValue *, 4>()) +
                 distDimsStr + ";\n";
  }
  if (port == hir::Details::w || port == hir::Details::rw) {
    out_decls += " wire " + strMemrefWrEn() + distDimsStr + ";\n";
    out_decls += "reg[" + to_string(dataWidth - 1) + ":0] " +
                 strMemrefWrData() + distDimsStr + ";\n";
  }
  return out_decls;
}

string VerilogValue::strMemrefArgDecl() {
  string out;
  MemrefType memrefTy = type.dyn_cast<MemrefType>();
  hir::Details::PortKind port = memrefTy.getPort();
  string portString =
      ((port == hir::Details::r) ? "r"
                                 : (port == hir::Details::w) ? "w" : "rw");
  out += "//MemrefType : port = " + portString + ".\n";
  unsigned addrWidth = helper::calcAddrWidth(memrefTy);
  string distDimsStr = strMemrefDistDims();
  bool printComma = false;
  unsigned dataWidth = helper::getBitWidth(memrefTy.getElementType());
  if (addrWidth > 0) { // all dims may be distributed.
    out += "output reg[" + to_string(addrWidth - 1) + ":0] " + strMemrefAddr() +
           distDimsStr;
    printComma = true;
  }
  if (port == hir::Details::r || port == hir::Details::rw) {
    if (printComma)
      out += ",\n";
    out += "output wire " + strMemrefRdEn() + distDimsStr;
    out += ",\ninput wire[" + to_string(dataWidth - 1) + ":0] " +
           strMemrefRdData() + distDimsStr;
    printComma = true;
  }
  if (port == hir::Details::w || port == hir::Details::rw) {
    if (printComma)
      out += ",\n";
    out += "output wire " + strMemrefWrEn() + distDimsStr;
    out += ",\noutput reg[" + to_string(dataWidth - 1) + ":0] " +
           strMemrefWrData() + distDimsStr;
  }
  return out;
}
#endif
