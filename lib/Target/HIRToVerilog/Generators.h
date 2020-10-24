#include "VerilogValue.h"
#include <string>
template <typename T>
bool isEqual(ArrayRef<T> x, ArrayRef<T> y) {
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
  string out;
  assert(ra.getPacking().size() >
         0); // if all dims are dist then it can't be bram.
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
  out +=
      "$name#(.SIZE($size), .WIDTH($width))(\nclk,\nclk,\n$ena,\n$enb,\n$wea,"
      "\n$web,\n$addra,\n"
      "$addrb,\n$dia,\n$dib,\n$doa,\n$dob\n);\n";
  for (auto dim : distDims) {
    out += "end\n";
  }

  string size = to_string(ra.getMemrefPackedSize());
  string width = to_string(getBitWidth(ra.getElementType()));
  string ena = ra.strMemrefRdEn() + dimSel;
  string enb = rb.strMemrefWrEn() + dimSel;
  string wea = "0"; // ra is 'r' port. FIXME
  string web = enb + dimSel;
  string addra = ra.strMemrefAddr() + dimSel;
  string addrb = rb.strMemrefAddr() + dimSel;
  string dia = "0";
  string dib = rb.strMemrefWrData() + dimSel;
  string doa = ra.strMemrefRdData() + dimSel;
  string dob = "/*ignored*/";

  findAndReplaceAll(out, "$name", name);
  findAndReplaceAll(out, "$size", size);
  findAndReplaceAll(out, "$width", width);
  findAndReplaceAll(out, "$ena", ena);
  findAndReplaceAll(out, "$enb", enb);
  findAndReplaceAll(out, "$wea", wea);
  findAndReplaceAll(out, "$web", web);
  findAndReplaceAll(out, "$addra", addra);
  findAndReplaceAll(out, "$addrb", addrb);
  findAndReplaceAll(out, "$dia", dia);
  findAndReplaceAll(out, "$dib", dib);
  findAndReplaceAll(out, "$doa", doa);
  findAndReplaceAll(out, "$dob", dob);
  return out;
}
