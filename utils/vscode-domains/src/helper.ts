import {ChildProcessWithoutNullStreams, spawn} from 'child_process';

type Pending = {
  resolve: (value: any) => void; reject : (reason: Error) => void
};

export interface Summary {
  complete: boolean;
  domains: number;
  modules: number;
  values: number;
  instances: number;
  provenanceEdges: number;
  skippedEdges: number;
  illegalCrossings?: number;
  hasIllegalCrossings?: boolean;
}

/** A single report process. Every protocol message is one JSON line. */
export class Helper {
  private readonly process: ChildProcessWithoutNullStreams;
  private readonly pending = new Map<number, Pending>();
  private nextId = 1;
  private partialLine = '';
  private readyResolve!: (summary: Summary) => void;
  private readyReject!: (reason: Error) => void;
  private readonly readyPromise: Promise<Summary>;
  private ready = false;
  private disposed = false;
  private failure?: Error;
  private stderr = '';

  constructor(executable: string, report: string,
              private readonly progress: (bytes: number, total: number) => void,
              private readonly log: (message: string) => void) {
    this.readyPromise = new Promise<Summary>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    this.process = spawn(executable, [ report ]);
    this.process.stdout.setEncoding('utf8');
    this.process.stderr.setEncoding('utf8');
    this.process.stdout.on('data', (chunk: string) => this.onData(chunk));
    this.process.stderr.on('data', (chunk: string) => {
      this.stderr += chunk;
      this.log(chunk);
    });
    this.process.on('error', (error) => this.fail(error));
    this.process.on('exit', (code, signal) => {
      const detail = this.stderr.trim();
      this.fail(new Error(`Domain report helper exited (${signal ?? code})${
          detail ? `: ${detail}` : ''}`));
    });
  }

  waitReady(): Promise<Summary> { return this.readyPromise; }

  request<T = any>(method: string, params: object = {}): Promise<T> {
    if (this.failure)
      return Promise.reject(this.failure);
    if (this.disposed)
      return Promise.reject(new Error('Domain report helper is closed'));
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      this.process.stdin.write(JSON.stringify({id, method, params}) + '\n',
                               (error) => {
                                 if (error) {
                                   this.pending.delete(id);
                                   reject(error);
                                 }
                               });
    });
  }

  dispose(): void {
    if (this.disposed)
      return;
    this.disposed = true;
    this.process.kill();
    this.fail(new Error('Domain report helper closed'));
  }

  private onData(chunk: string): void {
    this.partialLine += chunk;
    while (true) {
      const newline = this.partialLine.indexOf('\n');
      if (newline < 0)
        return;
      const line = this.partialLine.slice(0, newline);
      this.partialLine = this.partialLine.slice(newline + 1);
      if (!line.trim())
        continue;
      let message: any;
      try {
        message = JSON.parse(line);
      } catch (error) {
        this.fail(new Error(`Invalid helper response: ${String(error)}`));
        return;
      }
      if (message.event === 'progress') {
        this.progress(message.bytes, message.total);
      } else if (message.event === 'ready') {
        this.ready = true;
        this.readyResolve(message.summary as Summary);
      } else if (message.event === 'error') {
        this.fail(new Error(message.message));
      } else if (message.event === 'protocolError') {
        this.log(`Protocol error: ${message.message}`);
      } else if (typeof message.id === 'number') {
        const pending = this.pending.get(message.id);
        if (pending) {
          this.pending.delete(message.id);
          if (typeof message.error === 'string')
            pending.reject(new Error(message.error));
          else
            pending.resolve(message.result);
        }
      }
    }
  }

  private fail(error: Error): void {
    if (this.failure)
      return;
    this.failure = error;
    if (!this.ready)
      this.readyReject(error);
    for (const pending of this.pending.values())
      pending.reject(error);
    this.pending.clear();
  }
}
