#include "VerilogValue.h"
#include "helper.h"

SmallVector<unsigned, 4> VerilogValue::getMemrefPackedDims() const {
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
  for (int i = 0; i < numInputs; i++) {
    out += "assign $str_en_input_if[" + to_string(i) +
           "]$distDimAccessStr = $str_en_input $distDimAccessStr [" +
           to_string(i) + "];\n";
  }
  // Print end/endgenerate.
  for (int i = 0; i < distDims.size(); i++) {
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
  for (int i = 0; i < numDistDims; i++) {
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
  else
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
  for (int i = 0; i < distDims.size(); i++) {
    string str_i = "i" + to_string(i);
    string str_dim = to_string(distDims[i]);
    out += "for(genvar " + str_i + " = 0; " + str_i + " < " + str_dim + ";" +
           str_i + "=" + str_i + " + 1) begin\n";
    distDimAccessStr += "[" + str_i + "]";
  }
  out += "assign $str_v $distDimAccessStr = $str_v_array $distDimAccessStr [" +
         to_string(idx) + "]";
  // Print end/endgenerate.
  for (int i = 0; i < distDims.size(); i++) {
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
  for (int i = 0; i < numInputs; i++) {
    out += "assign $v_valid_if[" + to_string(i) +
           "] $distDimAccessStr = $v_valid $distDimAccessStr [" + to_string(i) +
           "];\n";
    out += "assign $v_input_if[" + to_string(i) +
           "] $distDimAccessStr = $v_input $distDimAccessStr [" + to_string(i) +
           "];\n";
  }
  // Print end/endgenerate.
  for (int i = 0; i < distDims.size(); i++) {
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
  auto packing = type.dyn_cast<hir::MemrefType>().getPacking();
  for (int i = 0; i < distDimPos.size(); i++) {
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

void VerilogValue::setIntegerConst(int value) {
  isConstValue = true;
  constValue.val_int = value;
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
  return constValue.val_int;
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
    else
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
  } else {
    string out;
    if (getBitWidth() > 1)
      out += "[" + to_string(getBitWidth() - 1) + ":0] ";
    return out + strWire();
  }
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
