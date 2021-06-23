#ifndef __HIRVERILOGVALUE__
#define __HIRVERILOGVALUE__

#include "circt/Dialect/HIR/ExportVerilog/HIRToVerilog.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <string>

using namespace circt;
using namespace hir;
using namespace std;

class VerilogValue {
public:
  VerilogValue();
  VerilogValue(Value value, string name);

  ~VerilogValue() { initialized = false; }
  void reset() { initialized = false; }
  bool isInitialized() { return initialized; }

public:
  int numReads(ArrayRef<VerilogValue *> addr);
  int numWrites(ArrayRef<VerilogValue *> addr);
  unsigned numAccess(unsigned pos);
  unsigned numAccess(ArrayRef<VerilogValue *> addr);

public:
  unsigned maxNumReads();
  unsigned maxNumWrites();
  unsigned maxNumAccess();

  string strMemrefArgDecl();
  string strMemrefInstDecl() const;
  string strGroupArgDecl();
  string strArrayArgDecl();

  Type getType();
  ArrayRef<int64_t> getShape() const;
  SmallVector<int, 4> getPackedDims() const;
  Type getElementType() const;

private:
  string strMemrefDistDims() const;

  string strWireInput();
  string strMemrefAddrValid() const;
  string strMemrefAddrInput();
  string strMemrefWrDataValid();
  string strMemrefWrDataInput();
  string strMemrefRdEnInput();
  string strMemrefWrEnInput();

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
  SmallVector<unsigned, 4> getMemrefPackedDims() const;

public:
  unsigned getMemrefPackedSize() const;

  SmallVector<unsigned, 4> getMemrefDistDims() const;
  void setIntegerConst(int value);
  bool isIntegerConst() const;
  int getIntegerConst() const;

  string strMemrefSelDecl();
  string strWire() const;
  unsigned getBitWidth() const;

  bool isIntegerType() const;
  bool isFloatType() const;

  /// Checks if the type is implemented as a verilog wire or an array of
  /// wires.
  bool isSimpleType() const;

  string strWireDecl() const;
  string strConstOrError(int n = 0) const;

  string strConstOrWire(unsigned bitwidth) const;
  string strConstOrWire() const;

  string strWireValid();
  string strWireInput(unsigned idx);
  string strDelayedWire() const;
  string strDelayedWire(unsigned delay);
  string strMemrefAddr() const;
  string strMemrefAddrValidIf(unsigned idx);
  string strMemrefAddrValid(unsigned idx);
  string strMemrefAddrValid(ArrayRef<VerilogValue *> addr, unsigned idx) const;
  string strMemrefAddrInputIf(unsigned idx);
  string strMemrefAddrInput(unsigned idx);
  string strMemrefAddrInput(ArrayRef<VerilogValue *> addr, unsigned idx);

  string strMemrefRdData() const;
  string strMemrefRdData(ArrayRef<VerilogValue *> addr) const;
  string strMemrefWrData() const;
  string strMemrefWrDataValidIf(unsigned idx);
  string strMemrefWrDataValid(ArrayRef<VerilogValue *> addr, unsigned idx);
  string strMemrefWrDataInputIf(unsigned idx);
  string strMemrefWrDataInput(unsigned idx);
  string strMemrefWrDataInput(ArrayRef<VerilogValue *> addr, unsigned idx);
  string strMemrefRdEn() const;
  string strMemrefRdEnInputIf(unsigned idx);

  string strMemrefRdEnInput(unsigned idx);
  string strMemrefRdEnInput(ArrayRef<VerilogValue *> addr, unsigned idx);
  string strMemrefWrEn() const;
  string strMemrefWrEnInputIf(unsigned idx);

  string strMemrefWrEnInput(unsigned idx);
  string strMemrefWrEnInput(ArrayRef<VerilogValue *> addr, unsigned idx);
  void incMemrefNumReads();

  void incMemrefNumWrites();
  void incMemrefNumReads(ArrayRef<VerilogValue *> addr);
  void incMemrefNumWrites(ArrayRef<VerilogValue *> addr);

  int getMaxDelay() const;

private:
  void updateMaxDelay(int delay);

  bool isConstValue = false;
  union {
    int valInt;
  } constValue;

  int maxDelay = 0;

private:
  bool initialized = false;
  Value value;
  Type type;
  string name;
  SmallVector<unsigned, 4> distDims;
  SmallVector<unsigned, 4> distDimPos;
  struct {
    SmallVector<int, 1> numReads;
    SmallVector<int, 1> numWrites;
  } usageInfo;
};
#endif
