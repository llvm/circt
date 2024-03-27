from z3 import *
import numpy as np



# reachability unsat
# body = !(findMyFun(transitions->at(to_check).from, stateInvMap_fun)(solverVars->size(), solverVars->data()));
# s.add(forall(solverVars->at(solverVars->size()-1),  implies((solverVars->at(solverVars->size()-1)>=0 && solverVars->at(solverVars->size()-1)<time_bound), nestedForall(*solverVars,body,0))));

# reachability sat
# body = (findMyFun(transitions->at(to_check).from, stateInvMap_fun)(solverVars->size(), solverVars->data()));
# s.add(exists(solverVars->at(solverVars->size()-1),  implies((solverVars->at(solverVars->size()-1)>=0 && solverVars->at(solverVars->size()-1)<time_bound), nestedForall(*solverVars,body,0))));

