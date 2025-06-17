module {
  moore.module @simple_vectorization(out out : !moore.l4, in %in : !moore.l4) {
    %out = moore.net wire : <l4>
    
    %0 = moore.extract_ref %out from 3 : <l4> -> <l1>
    %1 = moore.extract %in from 3 : l4 -> l1
    moore.assign %0, %1 : l1
    
    %2 = moore.extract_ref %out from 2 : <l4> -> <l1>
    %3 = moore.extract %in from 2 : l4 -> l1
    moore.assign %2, %3 : l1
    
    %4 = moore.extract_ref %out from 1 : <l4> -> <l1>
    %5 = moore.extract %in from 1 : l4 -> l1
    moore.assign %4, %5 : l1
    
    %6 = moore.extract_ref %out from 0 : <l4> -> <l1>
    %7 = moore.extract %in from 0 : l4 -> l1
    moore.assign %6, %7 : l1
    
    %8 = moore.read %out : <l4>
    moore.output %8 : !moore.l4
  }
}