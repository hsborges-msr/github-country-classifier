/**
 * Run `task` over `items` with at most `concurrency` tasks in flight. The first failure stops scheduling and aborts the
 * signal passed to in-flight tasks; it is thrown once they settle.
 */
export async function forEachConcurrently<T>(items: readonly T[], concurrency: number, task: (item: T, signal: AbortSignal) => Promise<void>): Promise<void> {
  if (!Number.isSafeInteger(concurrency) || concurrency < 1) throw new RangeError("concurrency must be a positive safe integer");
  const failed = new AbortController();
  let next = 0;
  const worker = async (): Promise<void> => {
    while (!failed.signal.aborted && next < items.length) {
      const item = items[next++] as T;
      try {
        await task(item, failed.signal);
      } catch (error) {
        if (!failed.signal.aborted) failed.abort(error);
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  failed.signal.throwIfAborted();
}
