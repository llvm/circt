


#include "circt/Dialect/FIRRTL/FIRRTLFieldSource.h"

using namespace circt;
using namespace firrtl;



FieldSource::FieldSource(Operation* operation) {
    FModuleOp mod = cast<FModuleOp>(operation);
    for (auto port : mod.getBodyBlock()->getArguments) {

    }   
    for (auto& op : mod.getBodyBlock())
        visitOp(op);
}


void FieldSource::visitOp(Operation* op) {
    if (auto sf = dyn_cast<SubfieldOp>(op))
        visitSubfield(sf);
    else if (auto si = dyn_cast<SubindexOp>)
        visitSubindex(si);
else if (auto sa = dyn_cast<SubaccessOp>(op))
visitSubaccess(sa);
}

void FieldSource::visitSubfield(SubfieldOp sf) {

}

void FieldSource::visitSubindex(SubfieldOp sf) {
    
}

void FieldSource::visitSubaccess(SubfieldOp sf) {
   
}
