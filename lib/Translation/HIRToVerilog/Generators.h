#include "VerilogValue.h"
#include "helper.h"
#include <string>
// template <typename T>
// bool isEqual(ArrayRef<T> x, ArrayRef<T> y) {
//  if (x.size() != y.size()) {
//    fprintf(stderr, "%d != %d\n", x.size(), y.size());
//    return false;
//  }
//  for (int i = 0; i < x.size(); i++) {
//    if (x[i] != y[i]) {
//      fprintf(stderr, "i=%d, %d != %d\n", i, x[i], y[i]);
//      return false;
//    }
//  }
//  return true;
//}

template <typename T>
bool isEqual(T x, T y) {
  if (x.size() != y.size()) {
    fprintf(stderr, "%d != %d\n", x.size(), y.size());
    return false;
  }
  for (int i = 0; i < x.size(); i++) {
    if (x[i] != y[i]) {
      fprintf(stderr, "i=%d, %d != %d\n", i, x[i], y[i]);
      return false;
    }
  }
  return true;
}
std::string gen_bram(string name, const VerilogValue ra,
                     const VerilogValue rb) {
  static int bram_number = 0;
  string out;
  assert(isEqual(ra.getShape(), rb.getShape()));
  assert(isEqual(ra.getPacking(), rb.getPacking()));
  assert(ra.getElementType() == rb.getElementType());
  SmallVector<unsigned, 4> distDims = ra.getMemrefDistDims();
  int i = 0;
  string dimSel = "";
  for (auto dim : distDims) {
    string str_i = to_string(i);
    out += "for(genvar i" + str_i + "= 0; i" + str_i + "<" + to_string(dim) +
           ";i" + str_i + "+=1) begin\n";
    dimSel += "[i" + str_i + "]";
    i++;
  }
  if (ra.getPacking().size() == 0) {
    out += "always@(posedge clk) begin\n";
    out += "  if($web) $doa <= $dib;\n";
    out += "end\n";
  } else {
    out += "$name#(.SIZE($size), .WIDTH($width))  bram_inst" +
           to_string(bram_number++) +
           "(\n.clka(clk),\n.clkb(clk),\n.ena($ena),\n.enb($enb),\n.wea(0),"
           "\n.web($web),\n.addra($addra),\n"
           ".addrb($addrb),\n.dia(0),\n.dib($dib),\n.doa($doa),\n.dob("
           "/*ignored*/)\n);\n";
  }
  for (auto dim : distDims) {
    out += "end\n";
  }

  string size = to_string(ra.getMemrefPackedSize());
  string width = to_string(helper::getBitWidth(ra.getElementType()));
  string ena = ra.strMemrefRdEn() + dimSel;
  string enb = rb.strMemrefWrEn() + dimSel;
  string web = enb;
  string addra = ra.strMemrefAddr() + dimSel;
  string addrb = rb.strMemrefAddr() + dimSel;
  string dib = rb.strMemrefWrData() + dimSel;
  string doa = ra.strMemrefRdData() + dimSel;

  helper::findAndReplaceAll(out, "$name", name);
  helper::findAndReplaceAll(out, "$size", size);
  helper::findAndReplaceAll(out, "$width", width);
  helper::findAndReplaceAll(out, "$ena", ena);
  helper::findAndReplaceAll(out, "$enb", enb);
  helper::findAndReplaceAll(out, "$web", web);
  helper::findAndReplaceAll(out, "$addra", addra);
  helper::findAndReplaceAll(out, "$addrb", addrb);
  helper::findAndReplaceAll(out, "$dib", dib);
  helper::findAndReplaceAll(out, "$doa", doa);
  return out;
}
