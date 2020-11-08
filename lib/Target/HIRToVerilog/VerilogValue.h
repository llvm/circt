#ifndef __HIRVERILOGVALUE__
#define __HIRVERILOGVALUE__

#include "Helpers.h"
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Target/HIRToVerilog/HIRToVerilog.h"
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
  VerilogValue() : value(Value()), type(Type()), name("uninitialized") {}
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
  unsigned numAccess(ArrayRef<VerilogValue *> addr) {
    unsigned pos = posMemrefDistDimAccess(addr);
    return usageInfo.numReads[pos] + usageInfo.numWrites[pos];
  }
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
      // wires in array.
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
  string strStreamArgDecl();
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
  SmallVector<unsigned, 4> getMemrefPackedDims() const {
    SmallVector<unsigned, 4> out;
    auto memrefType = type.dyn_cast<hir::MemrefType>();
    auto shape = memrefType.getShape();
    auto packing = memrefType.getPacking();
    for (int i = 0; i < shape.size(); i++) {
      bool dimIsPacked = false;
      for (auto p : packing) {
        if (i == shape.size() - 1 - p)
          dimIsPacked = true;
      }
      if (dimIsPacked) {
        auto dim = shape[i];
        out.push_back(dim);
      }
    }
    return out;
  }

  string strMemrefDistDims() const {
    string out;
    SmallVector<unsigned, 4> distDims = getMemrefDistDims();
    for (auto dim : distDims) {
      out += "[" + to_string(dim - 1) + ":0]";
    }
    return out;
  }

  string strWireInput() { return name + "_input"; }
  string strMemrefAddrValid() const { return name + "_addr_valid"; }
  string strMemrefAddrInput() { return name + "_addr_input"; }
  string strMemrefWrDataValid() { return name + "_wr_data_valid"; }
  string strMemrefWrDataInput() { return name + "_wr_data_input"; }
  string strMemrefRdEnInput() { return name + "_rd_en_input"; }
  string strMemrefWrEnInput() { return name + "_wr_en_input"; }

  string buildEnableSelectorStr(string str_en, string str_en_input,
                                unsigned numInputs) {
    string out;
    auto distDims = getMemrefDistDims();
    string distDimsStr = strMemrefDistDims();
    string distDimAccessStr = "";

    // Define en_input signal.
    out += "wire [$numInputsMinus1:0] $str_en_input $distDimsStr;\n";

    // Print generate loops for distributed dimensions.
    if (!distDims.empty())
      out += "generate\n";
    for (int i = 0; i < distDims.size(); i++) {
      string str_i = "i" + to_string(i);
      string str_dim = to_string(distDims[i]);
      out += "for(genvar " + str_i + " = 0; " + str_i + " < " + str_dim + ";" +
             str_i + "=" + str_i + " + 1) begin\n";
      distDimAccessStr += "[" + str_i + "]";
    }

    // Assign the enable signal using en_input.
    out += "assign $str_en $distDimAccessStr =| $str_en_input "
           "$distDimAccessStr;\n";

    // Print end/endgenerate.
    for (int i = 0; i < distDims.size(); i++) {
      out += "end\n";
    }
    if (!distDims.empty())
      out += "endgenerate\n";

    findAndReplaceAll(out, "$numInputsMinus1", to_string(numInputs - 1));
    findAndReplaceAll(out, "$str_en_input", str_en_input);
    findAndReplaceAll(out, "$str_en", str_en);
    findAndReplaceAll(out, "$distDimAccessStr", distDimAccessStr);
    findAndReplaceAll(out, "$distDimsStr", distDimsStr);
    return out;
  }

  string buildDataSelectorStr(string str_v, string str_v_valid,
                              string str_v_input, unsigned numInputs,
                              unsigned dataWidth) {
    string out;
    auto distDims = getMemrefDistDims();
    string distDimsStr = strMemrefDistDims();
    string distDimAccessStr = "";

    // Define the valid and input wire arrays.
    out = "wire $v_valid $distDimsStr [$numInputsMinus1:0] ;\n"
          "wire [$dataWidthMinus1:0] $v_input $distDimsStr "
          "[$numInputsMinus1:0];\n ";

    // Print generate loops for distributed dimensions.
    if (!distDims.empty())
      out += "generate\n";
    for (int i = 0; i < distDims.size(); i++) {
      string str_i = "i" + to_string(i);
      string str_dim = to_string(distDims[i]);
      out += "for(genvar " + str_i + " = 0; " + str_i + " < " + str_dim + ";" +
             str_i + "=" + str_i + " + 1) begin\n";
      distDimAccessStr += "[" + str_i + "]";
    }

    // Assign the data bus($v) using valid and input arrays.
    out += "always@(*) begin\n"
           "if($v_valid_access[0] )\n$v = "
           "$v_input_access[0];\n";
    for (int i = 1; i < numInputs; i++) {
      out += "else if ($v_valid_access[" + to_string(i) + "])\n";
      out += "$v = $v_input_access[" + to_string(i) + "];\n";
    }
    out += "else\n $v = 'x;\n";
    out += "end\n";

    // Print end/endgenerate.
    for (int i = 0; i < distDims.size(); i++) {
      out += "end\n";
    }
    if (!distDims.empty())
      out += "endgenerate\n";

    findAndReplaceAll(out, "$v_valid_access", str_v_valid + distDimAccessStr);
    findAndReplaceAll(out, "$v_input_access", str_v_input + distDimAccessStr);
    findAndReplaceAll(out, "$v_valid", str_v_valid);
    findAndReplaceAll(out, "$v_input", str_v_input);
    findAndReplaceAll(out, "$v", str_v + distDimAccessStr);
    findAndReplaceAll(out, "$dataWidthMinus1", to_string(dataWidth - 1));
    findAndReplaceAll(out, "$dataWidth", to_string(dataWidth));
    findAndReplaceAll(out, "$numInputsMinus1", to_string(numInputs - 1));
    findAndReplaceAll(out, "$distDimsStr", distDimsStr);
    return out;
  }

  string strMemrefAddrValid(unsigned idx) {
    return strMemrefAddrValid() + "[" + to_string(idx) + "]";
  }
  string strMemrefAddrInput(unsigned idx) {
    return strMemrefAddrInput() + "[" + to_string(idx) + "]";
  }

  unsigned posMemrefDistDimAccess(ArrayRef<VerilogValue *> addr) const {
    assert(type.isa<hir::MemrefType>());
    unsigned out = 0;
    auto packing = type.dyn_cast<hir::MemrefType>().getPacking();
    for (int i = 0; i < distDimPos.size(); i++) {
      auto pos = distDimPos[i];
      int v = addr[pos]->getIntegerConst();
      out *= distDims[i];
      out += v;
    }
    return out;
  }

  string strMemrefDistDimAccess(ArrayRef<VerilogValue *> addr) const {
    assert(type.isa<hir::MemrefType>());
    string out;
    auto packing = type.dyn_cast<hir::MemrefType>().getPacking();
    for (int i = 0; i < addr.size(); i++) {
      bool isDistDim = true;
      for (auto p : packing) {
        if (p == addr.size() - 1 - i)
          isDistDim = false;
      }
      if (isDistDim) {
        VerilogValue *v = addr[i];
        out += "[" + v->strConstOrError() + "]";
      }
    }
    return out;
  }

public:
  unsigned getMemrefPackedSize() const {
    SmallVector<unsigned, 4> packedDims = getMemrefPackedDims();
    unsigned v = 1;
    for (auto dim : packedDims) {
      v *= dim;
    }
    return v;
  }

  SmallVector<unsigned, 4> getMemrefDistDims() const {
    SmallVector<unsigned, 4> out;
    auto memrefType = type.dyn_cast<hir::MemrefType>();
    auto shape = memrefType.getShape();
    auto packing = memrefType.getPacking();
    for (int i = 0; i < shape.size(); i++) {
      bool dimIsPacked = false;
      for (auto p : packing) {
        if (i == shape.size() - 1 - p)
          dimIsPacked = true;
      }
      if (!dimIsPacked)
        out.push_back(shape[i]);
    }
    return out;
  }
  void setIntegerConst(int value) {
    isConstValue = true;
    constValue.val_int = value;
    assert(getIntegerConst() == value);
  }

  bool isIntegerConst() const { return isConstValue && isIntegerType(); }
  int getIntegerConst() const {
    if (!isIntegerConst()) {
      emitError(value.getLoc(), "ERROR: Expected const.\n");
      assert(false);
    }
    return constValue.val_int;
  }
  string strMemrefSelDecl() {
    auto numAccess = maxNumAccess();
    if (numAccess == 0)
      return "//Unused memref " + strWire() + ".\n";
    stringstream output;
    auto str_addr = strMemrefAddr();
    // print addr bus selector.
    unsigned addrWidth = calcAddrWidth(type.dyn_cast<MemrefType>());
    if (addrWidth > 0) {
      output << buildDataSelectorStr(strMemrefAddr(), strMemrefAddrValid(),
                                     strMemrefAddrInput(), numAccess,
                                     addrWidth);
      output << "\n";
    }
    // print rd_en selector.
    if (maxNumReads() > 0) {
      output << buildEnableSelectorStr(strMemrefRdEn(), strMemrefRdEnInput(),
                                       maxNumReads());
      output << "\n";
    }

    // print write bus selector.
    if (maxNumWrites() > 0) {
      unsigned dataWidth =
          ::getBitWidth(type.dyn_cast<MemrefType>().getElementType());
      output << buildEnableSelectorStr(strMemrefWrEn(), strMemrefWrEnInput(),
                                       maxNumWrites());
      output << buildDataSelectorStr(strMemrefWrData(), strMemrefWrDataValid(),
                                     strMemrefWrDataInput(), maxNumWrites(),
                                     dataWidth);
      output << "\n";
    }
    return output.str();
  }

  string strWire() const {
    if (name == "") {
      return "/*ERROR: Anonymus variable.*/";
    }
    return name;
  }

  unsigned getBitWidth() const {
    if (isIntegerConst()) {
      int val = std::abs(getIntegerConst());
      if (val > 0)
        return std::ceil(std::log2(val + 1));
      else
        return 1;
    }
    return ::getBitWidth(type);
  }

  bool isIntegerType() const { return ::isIntegerType(type); }
  /// Checks if the type is implemented as a verilog wire or an array of wires.
  bool isSimpleType() const {
    if (isIntegerType())
      return true;
    else if (type.isa<hir::TimeType>())
      return true;
    return false;
  }
  string strWireDecl() const {
    assert(initialized);
    assert(name != "");
    assert(isSimpleType());
    if (auto wireType = type.dyn_cast<hir::WireType>()) {
      auto shape = wireType.getShape();
      auto elementType = wireType.getElementType();
      auto elementWidthStr = to_string(::getBitWidth(elementType) - 1) + ":0";
      string distDimsStr = "";
      for (auto dim : shape) {
        distDimsStr += "[" + to_string(dim - 1) + ":0]";
      }
      return "[" + elementWidthStr + "] " + strWire() + distDimsStr;
    } else {
      string out;
      if (getBitWidth() > 1)
        out += "[" + to_string(getBitWidth() - 1) + ":0] ";
      return out + strWire();
    }
  }
  string strConstOrError(int n = 0) const {
    if (isIntegerConst()) {
      string constStr = to_string(getIntegerConst() + n);
      if (name == "")
        return constStr;
      return "/*" + name + "=*/ " + constStr;
    }
    fprintf(stderr, "/*ERROR: Expected constant. Found %s + %d */",
            strWire().c_str(), n);
    assert(false);
  }

  string strConstOrWire(unsigned bitwidth) const {
    if (isIntegerConst()) {
      string vlogDecimalLiteral =
          to_string(bitwidth) + "'d" + to_string(getIntegerConst());
      if (name == "")
        return vlogDecimalLiteral;
      return "/*" + name + "=*/ " + vlogDecimalLiteral;
    }
    return strWire() + "[" + to_string(bitwidth - 1) + ":0]";
  }

  string strConstOrWire() const {
    if (isIntegerConst()) {
      string vlogDecimalLiteral =
          to_string(getBitWidth()) + "'d" + to_string(getIntegerConst());
      if (name == "")
        return vlogDecimalLiteral;
      return "/*" + name + "=*/ " + vlogDecimalLiteral;
    }
    return strWire();
  }

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
  string strMemrefAddrValid(ArrayRef<VerilogValue *> addr, unsigned idx) const {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefAddrValid() + distDimAccessStr + "[" + to_string(idx) + "]";
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
  string strMemrefWrDataValid(ArrayRef<VerilogValue *> addr, unsigned idx) {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefWrDataValid() + distDimAccessStr + "[" + to_string(idx) +
           "]";
  }
  string strMemrefWrDataInput(ArrayRef<VerilogValue *> addr, unsigned idx) {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefWrDataInput() + distDimAccessStr + "[" + to_string(idx) +
           "]";
  }

  string strMemrefRdEn() const { return name + "_rd_en"; }
  string strMemrefRdEnInput(ArrayRef<VerilogValue *> addr, unsigned idx) {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefRdEnInput() + distDimAccessStr + "[" + to_string(idx) + "]";
  }

  string strMemrefWrEn() const { return name + "_wr_en"; }
  string strMemrefWrEnInput(ArrayRef<VerilogValue *> addr, unsigned idx) {
    string distDimAccessStr = strMemrefDistDimAccess(addr);
    return strMemrefWrEnInput() + distDimAccessStr + "[" + to_string(idx) + "]";
  }
  void incMemrefNumReads(ArrayRef<VerilogValue *> addr) {
    unsigned pos = posMemrefDistDimAccess(addr);
    usageInfo.numReads[pos]++;
  }
  void incMemrefNumWrites(ArrayRef<VerilogValue *> addr) {
    unsigned pos = posMemrefDistDimAccess(addr);
    usageInfo.numWrites[pos]++;
  }

  string strStreamPop() const { return name + "_pop"; }
  string strStreamData() const { return name + "_data"; }
  string strStreamPush() const { return name + "_push"; }

  unsigned getMaxDelay() const { return maxDelay; }

private:
  void updateMaxDelay(unsigned delay) {
    maxDelay = (maxDelay > delay) ? maxDelay : delay;
  }
  bool isConstValue = false;
  union {
    int val_int;
  } constValue;

  unsigned maxDelay = 0;
};

string VerilogValue::strMemrefInstDecl() const {
  string out_decls;
  MemrefType memrefTy = type.dyn_cast<MemrefType>();
  hir::Details::PortKind port = memrefTy.getPort();
  string portString =
      ((port == hir::Details::r) ? "r"
                                 : (port == hir::Details::w) ? "w" : "rw");
  unsigned addrWidth = calcAddrWidth(memrefTy);
  string distDimsStr = strMemrefDistDims();
  unsigned dataWidth = ::getBitWidth(memrefTy.getElementType());
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

string VerilogValue::strStreamArgDecl() {
  string out;
  StreamType streamTy = type.dyn_cast<StreamType>();
  assert(streamTy);
  hir::Details::PortKind port = streamTy.getPort();
  string portString =
      ((port == hir::Details::r) ? "r"
                                 : (port == hir::Details::w) ? "w" : "rw");
  out += "//StreamType : port = " + portString + ".\n";
  unsigned dataWidth = ::getBitWidth(streamTy.getElementType());

  // Bidirectional streams do not make much sense.
  assert(port != hir::Details::rw);

  if (port == hir::Details::r) {
    out += "output wire " + strStreamPop();
    out +=
        ",\ninput wire[" + to_string(dataWidth - 1) + ":0] " + strStreamData();
  } else if (port == hir::Details::w) {
    out += "output wire " + strStreamPush();
    out +=
        ",\noutput wire[" + to_string(dataWidth - 1) + ":0] " + strStreamData();
  }
  return out;
}

string VerilogValue::strMemrefArgDecl() {
  string out;
  MemrefType memrefTy = type.dyn_cast<MemrefType>();
  hir::Details::PortKind port = memrefTy.getPort();
  string portString =
      ((port == hir::Details::r) ? "r"
                                 : (port == hir::Details::w) ? "w" : "rw");
  out += "//MemrefType : port = " + portString + ".\n";
  unsigned addrWidth = calcAddrWidth(memrefTy);
  string distDimsStr = strMemrefDistDims();
  bool printComma = false;
  unsigned dataWidth = ::getBitWidth(memrefTy.getElementType());
  if (addrWidth > 0) { // add dims may be distributed.
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
