#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
#ifndef HIRGEN
#define HIRGEN

class ExprTree {
public:
  class ExternalOp {
  public:
    ExternalOp(std::string opName, int offset, int opDelay, mlir::Type resultTy)
        : opName(opName), offset(offset), opDelay(opDelay), resultTy(resultTy) {
    }
    int getOffset() { return offset; }
    mlir::StringRef getOpName() { return opName; }
    int getOpDelay() { return opDelay; }
    mlir::Type getResultType() { return resultTy; }

  private:
    std::string opName;
    int offset;
    int opDelay;
    mlir::Type resultTy;
  };

  class BinOp : public ExternalOp {
  public:
    BinOp(std::string opName, ExprTree *left, ExprTree *right, int offset,
          int opDelay, mlir::Type resultTy)
        : ExternalOp(opName, offset, opDelay, resultTy), left(left),
          right(right) {}

    ExprTree *getLeftOperand() { return left; }
    ExprTree *getRightOperand() { return right; }

  private:
    ExprTree *left;
    ExprTree *right;
  };

  class UnaryOp : public ExternalOp {
  public:
    UnaryOp(std::string opName, ExprTree *input, int offset, int opDelay,
            mlir::Type resultTy)
        : ExternalOp(opName, offset, opDelay, resultTy), input(input) {}
    ExprTree *getInputOperand() { return input; }

  private:
    ExprTree *input;
  };

  class SelectOp {
  public:
    SelectOp(ExprTree *input, mlir::SmallVector<int, 4> indices,
             mlir::Type resultTy)
        : input(input), indices(indices), resultTy(resultTy) {}
    mlir::SmallVectorImpl<int> &getIndices() { return indices; }
    mlir::Type getResultType() { return resultTy; }
    ExprTree *getInputOperand() { return input; }

  private:
    ExprTree *input;
    mlir::SmallVector<int, 4> indices;
    mlir::Type resultTy;
  };

public:
  ExprTree(int valueIdx) {
    tag = Tag::valueIdx;
    item.valueIdx = valueIdx;
  }
  ExprTree(UnaryOp *unaryOp) {
    tag = Tag::unaryOp;
    item.unaryOp = unaryOp;
  }
  ExprTree(BinOp *binOp) {
    tag = Tag::binOp;
    item.binOp = binOp;
  }
  ExprTree(SelectOp *selectOp) {
    tag = Tag::selectOp;
    item.selectOp = selectOp;
  }

  mlir::Value getValue(mlir::SmallVectorImpl<mlir::Value> &args) {
    if (tag == valueIdx)
      return args[item.valueIdx];
    return mlir::Value();
  }

  BinOp *getBinOp() {
    if (tag == binOp)
      return item.binOp;
    return NULL;
  }

  UnaryOp *getUnaryOp() {
    if (tag == unaryOp)
      return item.unaryOp;
    return NULL;
  }

  SelectOp *getSelectOp() {
    if (tag == selectOp)
      return item.selectOp;
    return NULL;
  }

  mlir::Type getResultType(mlir::SmallVectorImpl<mlir::Type> &argTypes) {
    if (tag == unaryOp)
      return item.unaryOp->getResultType();
    if (tag == binOp)
      return item.binOp->getResultType();
    if (tag == selectOp)
      return item.selectOp->getResultType();
    assert(item.valueIdx < (int)argTypes.size());
    return argTypes[item.valueIdx];
  }
  int getResultOffset() {

    if (tag == unaryOp)
      return item.unaryOp->getOffset() + item.unaryOp->getOpDelay();
    if (tag == binOp)
      return item.binOp->getOffset() + item.binOp->getOpDelay();
    return 0;
  }

private:
  enum Tag { valueIdx, unaryOp, binOp, selectOp } tag;
  union {
    int valueIdx;
    UnaryOp *unaryOp;
    BinOp *binOp;
    SelectOp *selectOp;
  } item;
};

int emitMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module);

void emitLineBuffer(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                    mlir::OwningModuleRef &module, mlir::StringRef funcName,
                    int imgDims[2], int kernelDims[2]);

void emitSum2D(mlir::OpBuilder &builder, mlir::MLIRContext &context,
               mlir::OwningModuleRef &module, int dims[2]);
void emitDet(mlir::OpBuilder &builder, mlir::MLIRContext &context,
             mlir::OwningModuleRef &module);
void emitTrace(mlir::OpBuilder &builder, mlir::MLIRContext &context,
               mlir::OwningModuleRef &module);
void emitHarris(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                mlir::OwningModuleRef &module);

void emitExpr(mlir::OpBuilder &builder, mlir::MLIRContext &context,
              mlir::OwningModuleRef &module, mlir::StringRef funcName,
              mlir::SmallVectorImpl<mlir::Type> &argTypes, ExprTree *expr);
#endif
