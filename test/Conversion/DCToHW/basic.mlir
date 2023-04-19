dc.func @simple(%0 : !dc.token, %1 : !dc.value<i64>, %2 : i1, %3 : !dc.value<i1, i2>)
        -> (!dc.token, !dc.value<i64>, i1, !dc.value<i1, i2>) {
    dc.return %0, %1, %2, %3 : !dc.token, !dc.value<i64>, i1, !dc.value<i1, i2>
}

dc.func @pack(%token : !dc.token, %v1 : i64, %v2 : i1) -> (!dc.value<i64, i1>) {
    %out = dc.pack %token [%v1, %v2] : (i64, i1) -> !dc.value<i64, i1>
    dc.return %out : !dc.value<i64, i1>
}

dc.func @unpack(%v : !dc.value<i64, i1>) -> (!dc.token, i64, i1) {
    %out:3 = dc.unpack %v : (!dc.value<i64, i1>) -> (i64, i1)
    dc.return %out#0, %out#1, %out#2 : !dc.token, i64, i1
}

dc.func @join(%t1 : !dc.token, %t2 : !dc.token) -> (!dc.token) {
    %out = dc.join %t1, %t2
    dc.return %out : !dc.token
}

dc.func @fork(%t : !dc.token) -> (!dc.token, !dc.token) {
    %out:2 = dc.fork %t : !dc.token, !dc.token
    dc.return %out#0, %out#1 : !dc.token, !dc.token
}

dc.func @bufferToken(%t1 : !dc.token) -> (!dc.token) {
    %out = dc.buffer %t1 [2] : !dc.token
    dc.return %out : !dc.token
}

dc.func @bufferValue(%v1 : !dc.value<i64>) -> (!dc.value<i64>) {
    %out = dc.buffer %v1 [2] : !dc.value<i64>
    dc.return %out : !dc.value<i64>
}

dc.func @branch(%sel : !dc.value<i1>, %token : !dc.token) -> (!dc.token, !dc.token) {
    // Canonicalize away a merge that is fed by a branch with the same select
    // input.
    %true, %false = dc.branch %sel, %token
    dc.return %true, %false : !dc.token, !dc.token
}

dc.func @merge(%sel : !dc.value<i1>, %true : !dc.token, %false : !dc.token) -> (!dc.token) {
    %0 = dc.merge %sel [ %true, %false ] : !dc.value<i1>
    dc.return %0 : !dc.token
}
