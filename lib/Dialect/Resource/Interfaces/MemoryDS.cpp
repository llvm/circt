#include "circt/Dialect/Resource/Interfaces/MemoryDS.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

using namespace llvm;
using namespace mlir;

namespace circt::hls_analysis {

llvm::DenseMap<StringRef, PartitionSpec>
getPartitionSpecs(Operation* op) {
  
  llvm::DenseMap<StringRef, PartitionSpec> specs;
  if (!op)
    return specs;

  auto arr = op->getAttrOfType<ArrayAttr>("hls.array_partition");
  if (!arr)
    return specs;

  for (Attribute a : arr) {
    auto dict = llvm::dyn_cast<DictionaryAttr>(a);
    if (!dict)
      continue;

    auto dimAttr = dict.getAs<IntegerAttr>("dim");
    auto factorAttr = dict.getAs<IntegerAttr>("factor");
    auto kindAttr = dict.getAs<StringAttr>("kind");
    auto nameAttr = dict.getAs<StringAttr>("variable");
    if (!dimAttr || !kindAttr || !nameAttr) {
      llvm::outs() << "Array partition missing keys\n";
      continue;
    }
    
    unsigned hlsDim = dimAttr.getInt(); // 1-indexed
    if (hlsDim == 0)
      continue; // dim=0 means "all dims" in HLS; handle if you need it
    unsigned d = hlsDim - 1; // -> MLIR 0-indexed

    PartitionSpec s;
    StringRef kind = kindAttr.getValue();
    if (kind == "cyclic")
      s.kind = PartitionSpec::Cyclic;
    else if (kind == "block")
      s.kind = PartitionSpec::Block;
    else
      s.kind = PartitionSpec::Complete;
    // complete partitioning: every element its own bank; factor irrelevant.
    s.factor = factorAttr ? factorAttr.getInt() : 1;
    s.dim = d;
    specs[nameAttr] = s;
  }
  return specs;
}
}