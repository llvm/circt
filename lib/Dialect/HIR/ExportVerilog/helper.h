#ifndef __HIRToVerilogHELPER__
#define __HIRToVerilogHELPER__
#include "VerilogValue.h"
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Translation/HIRToVerilog.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/Support/raw_ostream.h"

#include <list>
#include <locale>
#include <stack>

using namespace circt;
using namespace hir;
using namespace std;

namespace helper {
void findAndReplaceAll(string &data, string toSearch, string replaceStr);
unsigned getBitWidth(Type type);
unsigned calcAddrWidth(hir::MemrefType memrefTy);

} // namespace helper

namespace helper {
/// Handles string replacements in the generated code.
class StringReplacerClass {
public:
  StringReplacerClass(stringstream &outBuffer) : outBuffer(outBuffer) {}
  string insert(function<string()> func) {
    unsigned loc = replaceLocs.size();
    replaceLocs.push_back(make_pair(loc, func));
    return "$loc" + to_string(loc);
  }
  void pushFrame() { topOfStack.push(replaceLocs.size()); }
  void popFrame() {
    processReplacements();
    topOfStack.pop();
  }

private:
  std::stringstream &outBuffer;
  void processReplacements() {
    string code = outBuffer.str();
    outBuffer.str(std::string());
    int numEntries = replaceLocs.size() - topOfStack.top();
    for (int i = 0; i < numEntries; i++) {
      auto replacement = replaceLocs.back();
      replaceLocs.pop_back();
      unsigned loc = replacement.first;
      function<string()> getReplacementString = replacement.second;
      string replacementStr = getReplacementString();
      stringstream locSStream;
      locSStream << "$loc" << loc;
      helper::findAndReplaceAll(code, locSStream.str(), replacementStr);
    }
    outBuffer << code;
    assert(replaceLocs.size() == topOfStack.top());
  }

  std::stack<unsigned> topOfStack;
  std::list<std::pair<unsigned, function<string()>>> replaceLocs;
};

class VerilogMapperClass {
private:
  llvm::DenseMap<Value, VerilogValue *> mapValueToVerilogValuePtr;
  std::list<VerilogValue> verilogValueStore;
  std::stack<unsigned> topOfStack;

public:
  void pushFrame() { topOfStack.push(verilogValueStore.size()); }
  unsigned size() { return verilogValueStore.size(); }
  void popFrame() {
    int numDeletes = verilogValueStore.size() - topOfStack.top();
    for (int i = 0; i < numDeletes; i++) {
      verilogValueStore.pop_back();
    }
    topOfStack.pop();
  }

  VerilogValue *getMutable(Value v) {
    assert(v);
    if (mapValueToVerilogValuePtr.find(v) == mapValueToVerilogValuePtr.end()) {
      emitError(v.getDefiningOp()->getLoc(),
                "could not find VerilogValue for this mlir::Value!");
      assert(false);
    }
    VerilogValue *out = mapValueToVerilogValuePtr[v];
    assert(out->isInitialized());
    return out;
  }
  const VerilogValue get(Value v) { return *getMutable(v); }

  void insertPtr(Value v, VerilogValue *vv) {
    mapValueToVerilogValuePtr[v] = vv;
    assert(mapValueToVerilogValuePtr[v]->isInitialized());
  }

  void insert(Value v, VerilogValue vv) {
    verilogValueStore.push_back(vv);
    insertPtr(v, &verilogValueStore.back());
  }
  unsigned stackLevel() { return topOfStack.size(); }
};
} // namespace helper
#endif
