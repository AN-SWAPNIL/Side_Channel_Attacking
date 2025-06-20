/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;

function readNlines(n) {
  /*
   * Implement this function to read n cache lines.
   * 1. Allocate a buffer of size n * LINESIZE.
   * 2. Read each cache line (read the buffer in steps of LINESIZE) 10 times.
   * 3. Collect total time taken in an array using `performance.now()`.
   * 4. Return the median of the time taken in milliseconds.
   */

  const bufferSize = n * LINESIZE;
  if (bufferSize === 0) {
    return 0;
  }
  const buffer = new ArrayBuffer(bufferSize);
  const view = new Uint8Array(buffer);
  const times = [];

  console.log(
    `Buffer size: ${buffer.byteLength} bytes, n: ${n}, LINESIZE: ${LINESIZE} bytes`
  );

  for (let i = 0; i < 10; i++) {
    const startTime = performance.now();
    for (let j = 0; j < n; j++) {
      // Reading the first byte of each cache line
      let temp = view[j * LINESIZE];
    }
    const endTime = performance.now();
    times.push(endTime - startTime);
  }

  times.sort((a, b) => a - b);
  // Median of 10 elements is the average of 5th and 6th element
  const median = (times[4] + times[5]) / 2;
  return median;
}

self.addEventListener("message", function (e) {
  if (e.data === "start") {
    console.log("=== Task 1: Timing Warmup Challenge ===");
    console.log(
      "Measuring browser timing precision for side-channel attacks..."
    );
    console.log(`Cache Line Size: ${LINESIZE} bytes`);
    const results = {};

    /* Call the readNlines function for n = 1, 10, ... 10,000,000 and store the result */
    for (let n = 1; n <= 10000000; n *= 10) {
      console.log(`Testing n=${n} cache lines (${n * LINESIZE} bytes)`);
      const startTime = performance.now();
      results[n] = readNlines(n);
      const endTime = performance.now();
      console.log(
        `  Median time: ${results[n].toFixed(4)} ms (measurement took ${(
          endTime - startTime
        ).toFixed(2)} ms)`
      );

      // Break if we get unreliable measurements
      // if (results[n] === 0 || isNaN(results[n])) {
      //   console.log(`  Breaking due to unreliable measurement at n=${n}`);
      //   break;
      // }
    }

    console.log("=== Timing Analysis Complete ===");
    console.log("Results show browser timing resolution limits for security");
    self.postMessage(results);
  }
});
