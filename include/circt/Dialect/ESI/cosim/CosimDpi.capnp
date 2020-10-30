@0xe642127a31681ef6;

interface CosimDpiServer {
    list @0 () -> (ifaces :List(EsiDpiInterfaceDesc));
    open @1 [S, T] (iface :EsiDpiInterfaceDesc) -> (iface :EsiDpiEndpoint(S, T));
}

struct EsiDpiInterfaceDesc {
    typeID @0 :UInt64;
    endpointID @1 :Int32;
}

interface EsiDpiEndpoint(SendMsgType, RecvMsgType) {
    send @0 (msg :SendMsgType);
    recv @1 (block :Bool = true) -> (hasData :Bool, resp :RecvMsgType); # If 'resp' null, no data

    close @2 ();
}

struct UntypedData {
    data @0 :Data;
}