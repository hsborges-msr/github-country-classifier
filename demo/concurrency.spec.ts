import { expect, test } from "vitest";
import { forEachConcurrently } from "./concurrency.js";

const tick = () => new Promise(resolve => setTimeout(resolve, 1));

test("runs every item with at most `concurrency` tasks in flight", async () => {
  let running = 0;
  let peak = 0;
  const done: number[] = [];
  await forEachConcurrently([1, 2, 3, 4, 5, 6, 7], 3, async item => {
    running += 1;
    peak = Math.max(peak, running);
    await tick();
    done.push(item);
    running -= 1;
  });
  expect(peak).toBe(3);
  expect(done.sort()).toEqual([1, 2, 3, 4, 5, 6, 7]);
});

test("the first failure aborts the other tasks, starts no new ones and is thrown", async () => {
  const started: number[] = [];
  const aborted: number[] = [];
  const failure = new Error("rate limited");
  await expect(
    forEachConcurrently([1, 2, 3, 4, 5], 2, async (item, signal) => {
      started.push(item);
      if (item === 1) throw failure;
      await tick();
      if (signal.aborted) aborted.push(item);
    }),
  ).rejects.toBe(failure);
  expect(started).toEqual([1, 2]);
  expect(aborted).toEqual([2]);
});

test("rejects a concurrency below one", async () => {
  await expect(forEachConcurrently([1], 0, async () => {})).rejects.toThrow(RangeError);
});
