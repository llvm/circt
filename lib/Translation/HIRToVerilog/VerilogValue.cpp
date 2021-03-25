#include "VerilogValue.h"
#include "helper.h"

static bool isIntegerType(Type type) {
  assert(type);
  if (type.isa<IntegerType>())
    return true;
  if (type.isa<hir::ConstType>())
    return true;
  if (auto wireType = type.dyn_cast<WireType>())
    return isIntegerType(wireType.getElementType());
  return false;
}

VerilogValue::VerilogValue()
    : value(Value()), type(Type()), name("uninitialized") {
  maxDelay = 0;
}

VerilogValue::VerilogValue(Value value, string name)
    : initialized(true), value(value), type(value.getType()), name(name) {
  string out;
  auto size = 1;
  if (type.isa<MemrefType>()) {
    auto packing = type.dyn_cast<hir::MemrefType>().getPacking();
    auto shape = type.dyn_cast<hir::MemrefType>().getShape();
    for (size_t i = 0; i < shape.size(); i++) {
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

int VerilogValue::numReads(ArrayRef<VerilogValue *> addr) {
  unsigned pos = posMemrefDistDimAccess(addr);
  return usageInfo.numReads[pos];
}

int VerilogValue::numWrites(ArrayRef<VerilogValue *> addr) {
  unsigned pos = posMemrefDistDimAccess(addr);
  return usageInfo.numWrites[pos];
}
unsigned VerilogValue::numAccess(unsigned pos) {
  return usageInfo.numReads[pos] + usageInfo.numWrites[pos];
}
unsigned VerilogValue::numAccess(ArrayRef<VerilogValue *> addr) {
  unsigned pos = posMemrefDistDimAccess(addr);
  return numAccess(pos);
}
unsigned VerilogValue::maxNumReads() {
  unsigned out = 0;
  for (size_t i = 0; i < usageInfo.numReads.size(); i++) {
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
unsigned VerilogValue::maxNumWrites() {
  unsigned out = 0;
  for (size_t i = 0; i < usageInfo.numReads.size(); i++) {
    unsigned nWrites = usageInfo.numWrites[i];
    // TODO: We don't yet support different number of writes to different
    // wires in array. See printCallOp in HIRToVerilog. It increments
    // numReads for all reads.
    if (i != 0 && out != nWrites) {
      emitWarning(this->value.getLoc(),
                  "Number of writes is not same for all elements.");
    }
    out = (out > nWrites) ? out : nWrites;
  }
  return out;
}
unsigned VerilogValue::maxNumAccess() {
  unsigned out = 0;
  for (size_t i = 0; i < usageInfo.numReads.size(); i++) {
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
SmallVector<unsigned, 4> VerilogValue::getMemrefPackedDims() const {
  SmallVector<unsigned, 4> out;
  auto memrefType = type.dyn_cast<hir::MemrefType>();
  auto shape = memrefType.getShape();
  auto packing = memrefType.getPacking();
  for (size_t i = 0; i < shape.size(); i++) {
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

string VerilogValue::strMemrefDistDims() const {
  string out;
  SmallVector<unsigned, 4> distDims = getMemrefDistDims();
  for (auto dim : distDims) {
    out += "[" + to_string(dim - 1) + ":0]";
  }
  return out;
}

string VerilogValue::buildEnableSelectorStr(string str_en, string str_en_input,
                                            unsigned numInputs) {
  string out;
  auto distDims = getMemrefDistDims();
  string distDimsStr = strMemrefDistDims();
  string distDimAccessStr = "";

  // Define en_input signal.
  out += "wire [$numInputsMinus1:0] $str_en_input $distDimsStr;\n";
  out += "wire  $str_en_input_if [$numInputsMinus1:0] $distDimsStr;\n";

  // Print generate loops for distributed dimensions.
  if (!distDims.empty())
    out += "generate\n";
  for (size_t i = 0; i < distDims.size(); i++) {
    string str_i = "i" + to_string(i);
    string str_dim = to_string(distDims[i]);
    out += "for(genvar " + str_i + " = 0; " + str_i + " < " + str_dim + ";" +
           str_i + "=" + str_i + " + 1) begin\n";
    distDimAccessStr += "[" + str_i + "]";
  }

  // Assign the enable signal using en_input.
  out += "assign $str_en $distDimAccessStr =| $str_en_input "
         "$distDimAccessStr;\n";
  for (size_t i = 0; i < numInputs; i++) {
    out += "assign $str_en_input_if[" + to_string(i) +
           "]$distDimAccessStr = $str_en_input $distDimAccessStr [" +
           to_string(i) + "];\n";
  }
  // Print end/endgenerate.
  for (size_t i = 0; i < distDims.size(); i++) {
    out += "end\n";
  }
  if (!distDims.empty())
    out += "endgenerate\n";

  helper::findAndReplaceAll(out, "$numInputsMinus1", to_string(numInputs - 1));
  helper::findAndReplaceAll(out, "$str_en_input_if", str_en_input + "_if");
  helper::findAndReplaceAll(out, "$str_en_input", str_en_input);
  helper::findAndReplaceAll(out, "$str_en", str_en);
  helper::findAndReplaceAll(out, "$distDimAccessStr", distDimAccessStr);
  helper::findAndReplaceAll(out, "$distDimsStr", distDimsStr);
  return out;
}

string VerilogValue::strDistDimLoopNest(string distDimIdxStub, string body) {
  string prologue;
  string epilogue;
  auto distDims = getMemrefDistDims();
  string distDimAccessStr = "";
  unsigned numDistDims = distDims.size();
  // Print generate loops for distributed dimensions.
  for (size_t i = 0; i < numDistDims; i++) {
    string str_i = "i" + to_string(i);
    string str_dim = to_string(distDims[i]);
    prologue += "for(genvar " + str_i + " = 0; " + str_i + " < " + str_dim +
                ";" + str_i + "=" + str_i + " + 1) begin\n";
    epilogue += "end\n";
    distDimAccessStr += "[" + str_i + "]";
  }

  helper::findAndReplaceAll(body, distDimIdxStub, distDimAccessStr);
  if (numDistDims > 0)
    return "generate\n" + (prologue + body + epilogue) + "endgenerate\n";
  return (prologue + body + epilogue);
}

string VerilogValue::strMemrefSelect(string str_v, string str_v_array,
                                     unsigned idx, string str_dataWidth) {
  string out;
  auto distDims = getMemrefDistDims();
  string distDimsStr = strMemrefDistDims();
  string distDimAccessStr = "";

  // Define the .
  out = "wire $str_dataWidth $str_v $distDimsStr;\n";

  // Print generate loops for distributed dimensions.
  out += "generate\n";
  for (size_t i = 0; i < distDims.size(); i++) {
    string str_i = "i" + to_string(i);
    string str_dim = to_string(distDims[i]);
    out += "for(genvar " + str_i + " = 0; " + str_i + " < " + str_dim + ";" +
           str_i + "=" + str_i + " + 1) begin\n";
    distDimAccessStr += "[" + str_i + "]";
  }
  out += "assign $str_v $distDimAccessStr = $str_v_array $distDimAccessStr [" +
         to_string(idx) + "]";
  // Print end/endgenerate.
  for (size_t i = 0; i < distDims.size(); i++) {
    out += "end\n";
  }
  if (!distDims.empty())
    out += "endgenerate\n";

  helper::findAndReplaceAll(out, "$str_v", str_v);
  helper::findAndReplaceAll(out, "$str_v_array", str_v_array);
  helper::findAndReplaceAll(out, "$str_dataWidth", str_dataWidth);
  helper::findAndReplaceAll(out, "$distDimsStr", distDimsStr);
  helper::findAndReplaceAll(out, "$distDimAccessStr", distDimAccessStr);
  return out;
}

string VerilogValue::buildDataSelectorStr(string str_v, string str_v_valid,
                                          string str_v_input,
                                          unsigned numInputs,
                                          unsigned dataWidth) {
  string out;
  auto distDims = getMemrefDistDims();
  string distDimsStr = strMemrefDistDims();
  string distDimAccessStr = "";

  // Define the valid and input wire arrays.
  out = "wire $v_valid $distDimsStr [$numInputsMinus1:0] ;\n"
        "wire [$dataWidthMinus1:0] $v_input $distDimsStr "
        "[$numInputsMinus1:0];\n ";
  // Define the corresponding arrays with numInputs as first dimension. Used
  // by CallOp to select Whole array. "if" stands for "index first"
  out += "wire $v_valid_if [$numInputsMinus1:0] $distDimsStr  ;\n"
         "wire [$dataWidthMinus1:0] $v_input_if [$numInputsMinus1:0] "
         "$distDimsStr ;\n ";

  // Print generate loops for distributed dimensions.
  if (!distDims.empty())
    out += "generate\n";
  for (size_t i = 0; i < distDims.size(); i++) {
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
  for (size_t i = 1; i < numInputs; i++) {
    out += "else if ($v_valid_access[" + to_string(i) + "])\n";
    out += "$v = $v_input_access[" + to_string(i) + "];\n";
  }
  out += "else\n $v = 'x;\n";
  out += "end\n";
  for (size_t i = 0; i < numInputs; i++) {
    out += "assign $v_valid_if[" + to_string(i) +
           "] $distDimAccessStr = $v_valid $distDimAccessStr [" + to_string(i) +
           "];\n";
    out += "assign $v_input_if[" + to_string(i) +
           "] $distDimAccessStr = $v_input $distDimAccessStr [" + to_string(i) +
           "];\n";
  }
  // Print end/endgenerate.
  for (size_t i = 0; i < distDims.size(); i++) {
    out += "end\n";
  }
  if (!distDims.empty())
    out += "endgenerate\n";

  helper::findAndReplaceAll(out, "$v_valid_access",
                            str_v_valid + distDimAccessStr);
  helper::findAndReplaceAll(out, "$v_input_access",
                            str_v_input + distDimAccessStr);
  helper::findAndReplaceAll(out, "$v_valid_if", str_v_valid + "_if");
  helper::findAndReplaceAll(out, "$v_input_if", str_v_input + "_if");
  helper::findAndReplaceAll(out, "$v_valid", str_v_valid);
  helper::findAndReplaceAll(out, "$v_input", str_v_input);
  helper::findAndReplaceAll(out, "$v", str_v + distDimAccessStr);
  helper::findAndReplaceAll(out, "$dataWidthMinus1", to_string(dataWidth - 1));
  helper::findAndReplaceAll(out, "$dataWidth", to_string(dataWidth));
  helper::findAndReplaceAll(out, "$numInputsMinus1", to_string(numInputs - 1));
  helper::findAndReplaceAll(out, "$distDimsStr", distDimsStr);
  helper::findAndReplaceAll(out, "$distDimAccessStr", distDimAccessStr);
  return out;
}

unsigned
VerilogValue::posMemrefDistDimAccess(ArrayRef<VerilogValue *> addr) const {
  assert(type.isa<hir::MemrefType>());
  unsigned out = 0;
  for (size_t i = 0; i < distDimPos.size(); i++) {
    auto pos = distDimPos[i];
    int v = addr[pos]->getIntegerConst();
    out *= distDims[i];
    out += v;
  }
  return out;
}

string
VerilogValue::strMemrefDistDimAccess(ArrayRef<VerilogValue *> addr) const {
  assert(type.isa<hir::MemrefType>());
  string out;
  auto packing = type.dyn_cast<hir::MemrefType>().getPacking();
  for (size_t i = 0; i < addr.size(); i++) {
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

unsigned VerilogValue::getMemrefPackedSize() const {
  SmallVector<unsigned, 4> packedDims = getMemrefPackedDims();
  unsigned v = 1;
  for (auto dim : packedDims) {
    v *= dim;
  }
  return v;
}

SmallVector<unsigned, 4> VerilogValue::getMemrefDistDims() const {
  SmallVector<unsigned, 4> out;
  auto memrefType = type.dyn_cast<hir::MemrefType>();
  auto shape = memrefType.getShape();
  auto packing = memrefType.getPacking();
  for (size_t i = 0; i < shape.size(); i++) {
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

void VerilogValue::setIntegerConst(int value) {
  isConstValue = true;
  constValue.valInt = value;
  assert(getIntegerConst() == value);
}

bool VerilogValue::isIntegerConst() const {
  return isConstValue && isIntegerType();
}
int VerilogValue::getIntegerConst() const {
  if (!isIntegerConst()) {
    emitError(value.getLoc(), "ERROR: Expected const.\n");
    assert(false);
  }
  return constValue.valInt;
}
string VerilogValue::strMemrefSelDecl() {
  auto numAccess = maxNumAccess();
  if (numAccess == 0)
    return "//Unused memref " + strWire() + ".\n";
  stringstream output;
  auto strAddr = strMemrefAddr();
  // print addr bus selector.
  unsigned addrWidth = helper::calcAddrWidth(type.dyn_cast<MemrefType>());
  if (addrWidth > 0) {
    output << buildDataSelectorStr(strMemrefAddr(), strMemrefAddrValid(),
                                   strMemrefAddrInput(), numAccess, addrWidth);
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
        helper::getBitWidth(type.dyn_cast<MemrefType>().getElementType());
    output << buildEnableSelectorStr(strMemrefWrEn(), strMemrefWrEnInput(),
                                     maxNumWrites());
    output << buildDataSelectorStr(strMemrefWrData(), strMemrefWrDataValid(),
                                   strMemrefWrDataInput(), maxNumWrites(),
                                   dataWidth);
    output << "\n";
  }
  return output.str();
}

unsigned VerilogValue::getBitWidth() const {
  if (isIntegerConst()) {
    int val = std::abs(getIntegerConst());
    if (val > 0)
      return std::ceil(std::log2(val + 1));
    return 1;
  }
  return helper::getBitWidth(type);
}
bool VerilogValue::isIntegerType() const { return ::isIntegerType(type); }
bool VerilogValue::isFloatType() const { return type.isa<FloatType>(); }
string VerilogValue::strWireDecl() const {
  assert(initialized);
  assert(name != "");
  assert(isSimpleType());
  if (auto wireType = type.dyn_cast<hir::WireType>()) {
    auto shape = wireType.getShape();
    auto elementType = wireType.getElementType();
    auto elementWidthStr =
        to_string(helper::getBitWidth(elementType) - 1) + ":0";
    string distDimsStr = "";
    for (auto dim : shape) {
      distDimsStr += "[" + to_string(dim - 1) + ":0]";
    }
    return "[" + elementWidthStr + "] " + strWire() + distDimsStr;
  }
  // If not wire type.
  string out;
  if (getBitWidth() > 1)
    out += "[" + to_string(getBitWidth() - 1) + ":0] ";
  return out + strWire();
}

string VerilogValue::strConstOrError(int n) const {
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
string VerilogValue::strConstOrWire(unsigned bitwidth) const {
  if (isIntegerConst()) {
    string vlogDecimalLiteral =
        to_string(bitwidth) + "'d" + to_string(getIntegerConst());
    if (name == "")
      return vlogDecimalLiteral;
    return "/*" + name + "=*/ " + vlogDecimalLiteral;
  }
  return strWire() + "[" + to_string(bitwidth - 1) + ":0]";
}
string VerilogValue::strConstOrWire() const {
  if (isIntegerConst()) {
    string vlogDecimalLiteral =
        to_string(getBitWidth()) + "'d" + to_string(getIntegerConst());
    if (name == "")
      return vlogDecimalLiteral;
    return "/*" + name + "=*/ " + vlogDecimalLiteral;
  }
  return strWire();
}

string VerilogValue::strMemrefArgDecl() {
  string out;
  MemrefType memrefTy = type.dyn_cast<MemrefType>();
  hir::Details::PortKind port = memrefTy.getPort();
  string portString = ((port == hir::Details::r)   ? "r"
                       : (port == hir::Details::w) ? "w"
                                                   : "rw");
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

string VerilogValue::strMemrefInstDecl() const {
  string out_decls;
  MemrefType memrefTy = type.dyn_cast<MemrefType>();
  hir::Details::PortKind port = memrefTy.getPort();
  string portString = ((port == hir::Details::r)   ? "r"
                       : (port == hir::Details::w) ? "w"
                                                   : "rw");
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

Type VerilogValue::getType() { return type; }

ArrayRef<unsigned> VerilogValue::getShape() const {
  assert(type.isa<MemrefType>());
  return type.dyn_cast<MemrefType>().getShape();
}
ArrayRef<unsigned> VerilogValue::getPacking() const {
  assert(type.isa<MemrefType>());
  return type.dyn_cast<MemrefType>().getPacking();
}

Type VerilogValue::getElementType() const {
  assert(type.isa<MemrefType>());
  return type.dyn_cast<MemrefType>().getElementType();
}

string VerilogValue::strWireInput() { return name + "_input"; }
string VerilogValue::strMemrefAddrValid() const { return name + "_addr_valid"; }
string VerilogValue::strMemrefAddrInput() { return name + "_addr_input"; }
string VerilogValue::strMemrefWrDataValid() { return name + "_wr_data_valid"; }
string VerilogValue::strMemrefWrDataInput() { return name + "_wr_data_input"; }
string VerilogValue::strMemrefRdEnInput() { return name + "_rd_en_input"; }
string VerilogValue::strMemrefWrEnInput() { return name + "_wr_en_input"; }

string VerilogValue::strWire() const {
  if (name == "") {
    return "/*ERROR: Anonymus variable.*/";
  }
  return name;
}
bool VerilogValue::isSimpleType() const {
  if (isIntegerType() || isFloatType())
    return true;
  if (type.isa<hir::TimeType>())
    return true;
  return false;
}
string VerilogValue::strWireValid() { return name + "_valid"; }
string VerilogValue::strWireInput(unsigned idx) {
  return strWireInput() + "[" + to_string(idx) + "]";
}
string VerilogValue::strDelayedWire() const { return name + "delay"; }
string VerilogValue::strDelayedWire(unsigned delay) {
  updateMaxDelay(delay);
  return strDelayedWire() + "[" + to_string(delay) + "]";
}

string VerilogValue::strMemrefAddr() const { return name + "_addr"; }
string VerilogValue::strMemrefAddrValidIf(unsigned idx) {
  return strMemrefAddrValid() + "_if[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefAddrValid(unsigned idx) {
  return strMemrefAddrValid() + "[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefAddrValid(ArrayRef<VerilogValue *> addr,
                                        unsigned idx) const {
  string distDimAccessStr = strMemrefDistDimAccess(addr);
  return strMemrefAddrValid() + distDimAccessStr + "[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefAddrInputIf(unsigned idx) {
  return strMemrefAddrInput() + "_if[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefAddrInput(unsigned idx) {
  return strMemrefAddrInput() + "[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefAddrInput(ArrayRef<VerilogValue *> addr,
                                        unsigned idx) {
  string distDimAccessStr = strMemrefDistDimAccess(addr);
  return strMemrefAddrInput() + distDimAccessStr + "[" + to_string(idx) + "]";
}

string VerilogValue::strMemrefRdData() const { return name + "_rd_data"; }
string VerilogValue::strMemrefRdData(ArrayRef<VerilogValue *> addr) const {
  string distDimAccessStr = strMemrefDistDimAccess(addr);
  return strMemrefRdData() + distDimAccessStr;
}

string VerilogValue::strMemrefWrData() const { return name + "_wr_data"; }
string VerilogValue::strMemrefWrDataValidIf(unsigned idx) {
  return strMemrefWrDataValid() + "_if[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefWrDataValid(ArrayRef<VerilogValue *> addr,
                                          unsigned idx) {
  string distDimAccessStr = strMemrefDistDimAccess(addr);
  return strMemrefWrDataValid() + distDimAccessStr + "[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefWrDataInputIf(unsigned idx) {
  return strMemrefWrDataInput() + "_if[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefWrDataInput(unsigned idx) {
  return strMemrefWrDataInput() + "[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefWrDataInput(ArrayRef<VerilogValue *> addr,
                                          unsigned idx) {
  string distDimAccessStr = strMemrefDistDimAccess(addr);
  return strMemrefWrDataInput() + distDimAccessStr + "[" + to_string(idx) + "]";
}

string VerilogValue::strMemrefRdEn() const { return name + "_rd_en"; }
string VerilogValue::strMemrefRdEnInputIf(unsigned idx) {
  return strMemrefRdEnInput() + "_if[" + to_string(idx) + "]";
}

string VerilogValue::strMemrefRdEnInput(unsigned idx) {
  return strMemrefRdEnInput() + "[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefRdEnInput(ArrayRef<VerilogValue *> addr,
                                        unsigned idx) {
  string distDimAccessStr = strMemrefDistDimAccess(addr);
  return strMemrefRdEnInput() + distDimAccessStr + "[" + to_string(idx) + "]";
}

string VerilogValue::strMemrefWrEn() const { return name + "_wr_en"; }
string VerilogValue::strMemrefWrEnInputIf(unsigned idx) {
  return strMemrefWrEnInput() + "_if[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefWrEnInput(unsigned idx) {
  return strMemrefWrEnInput() + "[" + to_string(idx) + "]";
}
string VerilogValue::strMemrefWrEnInput(ArrayRef<VerilogValue *> addr,
                                        unsigned idx) {
  string distDimAccessStr = strMemrefDistDimAccess(addr);
  return strMemrefWrEnInput() + distDimAccessStr + "[" + to_string(idx) + "]";
}
void VerilogValue::incMemrefNumReads() {
  for (size_t pos = 0; pos < usageInfo.numReads.size(); pos++) {
    usageInfo.numReads[pos]++;
  }
}

void VerilogValue::incMemrefNumWrites() {
  for (size_t pos = 0; pos < usageInfo.numWrites.size(); pos++) {
    usageInfo.numWrites[pos]++;
  }
}
void VerilogValue::incMemrefNumReads(ArrayRef<VerilogValue *> addr) {
  unsigned pos = posMemrefDistDimAccess(addr);
  usageInfo.numReads[pos]++;
}
void VerilogValue::incMemrefNumWrites(ArrayRef<VerilogValue *> addr) {
  unsigned pos = posMemrefDistDimAccess(addr);
  usageInfo.numWrites[pos]++;
}

int VerilogValue::getMaxDelay() const {
  if (maxDelay < 0 || maxDelay > 64) {
    fprintf(stderr, "unexpected maxDelay %d\n", maxDelay);
  }
  return maxDelay;
}
void VerilogValue::updateMaxDelay(int delay) {
  if (delay < 0 || delay > 64) {
    fprintf(stderr, "unexpected delay %d\n", delay);
  }
  maxDelay = (maxDelay > delay) ? maxDelay : delay;
}
