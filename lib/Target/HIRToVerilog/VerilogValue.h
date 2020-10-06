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

class VerilogValue {
public:
  VerilogValue() : type(Type()), name("unknown") {}
  VerilogValue(Type type, string name) : type(type), name(name) {}

public:
  int numReads() { return usageInfo.numReads; }
  int numWrites() { return usageInfo.numWrites; }
  int numAccess() { return usageInfo.numReads + usageInfo.numWrites; }

private:
  SmallVector<unsigned, 4> getMemrefPackedDims() {
    SmallVector<unsigned, 4> out;
    auto memrefType = type.dyn_cast<hir::MemrefType>();
    auto shape = memrefType.getShape();
    auto packing = memrefType.getPacking();
    for (auto dim : shape) {
      bool dimIsPacked = false;
      for (auto packedDim : packing) {
        if (dim == packedDim)
          dimIsPacked = true;
      }
      if (dimIsPacked)
        out.push_back(dim);
    }
    return out;
  }

  SmallVector<unsigned, 4> getMemrefDistDims() {
    SmallVector<unsigned, 4> out;
    auto memrefType = type.dyn_cast<hir::MemrefType>();
    auto shape = memrefType.getShape();
    auto packing = memrefType.getPacking();
    for (int dim = 0; dim < shape.size(); dim++) {
      bool dimIsPacked = false;
      for (auto packedDim : packing) {
        if (dim == packedDim)
          dimIsPacked = true;
      }
      if (!dimIsPacked)
        out.push_back(shape[dim]);
    }
    return out;
  }

  string strWireInput() { return name + "_input"; }
  string strMemrefAddrValid() { return name + "_addr_valid"; }
  string strMemrefAddrInput() { return name + "_addr_input"; }
  string strMemrefWrDataValid() { return name + "_wr_data_valid"; }
  string strMemrefWrDataInput() { return name + "_wr_data_input"; }
  string strMemrefRdEnInput() { return name + "_rd_en_input"; }
  string strMemrefWrEnInput() { return name + "_wr_en_input"; }
  Type getType() { return type; }

  static string buildEnableSelectorStr(string &v_en, unsigned numInputs) {
    stringstream output;
    string v_en_input = v_en + "_input";
    output << "wire[" << numInputs - 1 << ":0]" << v_en_input << ";\n";
    output << "assign " << v_en << " = |" << v_en_input << ";\n";
    return output.str();
  }

  string buildDataSelectorStr(string str_v, string str_v_valid,
                              string str_v_input, unsigned numInputs,
                              unsigned dataWidth) {
    string out;
    auto distDims = this->getMemrefDistDims();
    string distDimsStr;
    for (auto dim : distDims) {
      distDimsStr += "[" + to_string(dim - 1) + ":0]";
    }
    out = "wire $distDimsStr $v_valid [$numInputsMinus1:0];\n"
          "wire $distDimsStr [$dataWidthMinus1:0] $v_input "
          "[$numInputsMinus1:0];\n ";

    if (!distDims.empty())
      out += "generate\n";
    string distDimAccessStr = "";
    for (int i = 0; i < distDims.size(); i++) {
      string str_i = to_string(i);
      string str_dim = to_string(distDims[i]);
      out += "genvar i" + str_i + ";\n";
      out += "for(i" + str_i + " = 0; i" + str_i + " < " + str_dim + ";i" +
             str_i + "++) begin\n";
      distDimAccessStr += "[i" + str_i + "]";
    }

    out += "always@(*) begin\n"
           "if($v_valid[0] $distDimAccessStr)\n$v = "
           "$v_input[0]$distDimAccessStr;\n";
    for (int i = 1; i < numInputs; i++) {
      out += "else if ($v_valid[" + to_string(i) + "]$distDimAccessStr)\n";
      out += "$v$distDimAccessStr = $v_input[" + to_string(i) +
             "]$distDimAccessStr;\n";
    }
    out += "else\n $v$distDimAccessStr = $dataWidth'd0;\n";
    out += "end\n";
    for (int i = 0; i < distDims.size(); i++) {
      out += "end\n";
    }
    if (!distDims.empty())
      out += "endgenerate\n";

    findAndReplaceAll(out, "$v", str_v);
    findAndReplaceAll(out, "$v_valid", str_v_valid);
    findAndReplaceAll(out, "$v_input", str_v_input);
    findAndReplaceAll(out, "$dataWidthMinus1", to_string(dataWidth - 1));
    findAndReplaceAll(out, "$dataWidth", to_string(dataWidth));
    findAndReplaceAll(out, "$numInputsMinus1", to_string(numInputs - 1));
    findAndReplaceAll(out, "$distDimsStr", distDimsStr);
    findAndReplaceAll(out, "$distDimAccessStr", distDimAccessStr);
    return out;
  }

public:
  string strMemrefDef() {
    if (this->numAccess() == 0)
      return "//Unused memref " + this->strWire() + ".\n";
    stringstream output;
    auto str_addr = this->strMemrefAddr();
    // print addr bus selector.
    unsigned addrWidth = calcAddrWidth(type.dyn_cast<MemrefType>());
    if (addrWidth > 0) {
      output << buildDataSelectorStr(
          this->strMemrefAddr(), this->strMemrefAddrInput(),
          this->strMemrefAddrValid(), this->numAccess(), addrWidth);
      output << "\n";
    }
    // print rd_en selector.
    if (this->numReads() > 0) {
      string v_rd_en = this->strMemrefRdEn();
      output << buildEnableSelectorStr(v_rd_en, this->numReads());
      output << "\n";
    }

    // print write bus selector.
    if (this->numWrites() > 0) {
      unsigned dataWidth = calcDataWidth(type.dyn_cast<MemrefType>());
      string str_wr_en = this->strMemrefWrEn();
      output << buildEnableSelectorStr(str_wr_en, this->numWrites());
      string str_wr_data = this->strMemrefWrData();
      output << buildDataSelectorStr(
          this->strMemrefWrData(), this->strMemrefWrDataInput(),
          this->strMemrefWrDataValid(), this->numWrites(), dataWidth);
      output << "\n";
    }
    return output.str();
  }

  string strWire() { return name; }
  string strWireValid() { return name + "_valid"; }
  string strWireInput(unsigned idx) {
    return strWireInput() + "[" + to_string(idx) + "]";
  }
  string strDelayedWire() { return name + "delay"; }
  string strDelayedWire(unsigned delay) {
    return strDelayedWire() + "[" + to_string(delay) + "]";
  }

  string strMemrefAddr() { return name + "_addr"; }
  string strMemrefAddrValid(unsigned idx) {
    return strMemrefAddrValid() + "[" + to_string(idx) + "]";
  }
  string strMemrefAddrInput(unsigned idx) {
    return strMemrefAddrInput() + "[" + to_string(idx) + "]";
  }

  string strMemrefRdData() { return name + "_rd_data"; }

  string strMemrefWrData() { return name + "_wr_data"; }
  string strMemrefWrDataValid(unsigned idx) {
    return strMemrefWrDataValid() + "[" + to_string(idx) + "]";
  }
  string strMemrefWrDataInput(unsigned idx) {
    return strMemrefWrDataInput() + "[" + to_string(idx) + "]";
  }

  string strMemrefRdEn() { return name + "_rd_en"; }
  string strMemrefRdEnInput(unsigned idx) {
    return strMemrefRdEnInput() + "[" + to_string(idx) + "]";
  }

  string strMemrefWrEn() { return name + "_wr_en"; }
  string strMemrefWrEnInput(unsigned idx) {
    return strMemrefWrEnInput() + "[" + to_string(idx) + "]";
  }
  void incNumReads() { usageInfo.numReads++; }
  void incNumWrites() { usageInfo.numWrites++; }

private:
  struct {
    unsigned numReads = 0;
    unsigned numWrites = 0;
  } usageInfo;
  Type type;
  string name;
};
#endif
