function app() {
  return {
    /* This is the main app object containing all the application state and methods. */
    // The following properties are used to store the state of the application

    // results of cache latency measurements
    latencyResults: null,
    // local collection of trace data
    traceData: [],
    // Local collection of heapmap images
    heatmaps: [],

    // Current status message
    status: "",
    // Is any worker running?
    isCollecting: false,
    // Is the status message an error?
    statusIsError: false,
    // Show trace data in the UI?
    showingTraces: false,

    // Helper function to get color based on probability
    getProbabilityColor(probability) {
      // Create a simple color gradient from red (0%) to green (100%)
      if (probability > 80) return "#28a745"; // green for high probability
      if (probability > 40) return "#007bff"; // blue for medium probability
      if (probability > 20) return "#ffc107"; // yellow for low-medium probability
      return "#dc3545"; // red for low probability
    },

    // Collect latency data using warmup.js worker
    async collectLatencyData() {
      this.isCollecting = true;
      this.status = "Collecting latency data...";
      this.latencyResults = null;
      this.statusIsError = false;
      this.showingTraces = false;

      try {
        // Create a worker
        let worker = new Worker("warmup.js");

        // Start the measurement and wait for result
        const results = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        // Update results
        this.latencyResults = results;
        this.status = "Latency data collection complete!";

        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting latency data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Collect trace data using worker.js and send to backend
    async collectTraceData() {
      this.isCollecting = true;
      this.status = "Collecting trace data...";
      this.statusIsError = false;
      // Don't hide existing traces during collection - keep them visible

      try {
        // Create a worker
        let worker = new Worker("worker.js");

        // Start the measurement and wait for result
        const result = await new Promise((resolve, reject) => {
          worker.onmessage = (e) => {
            if (e.data.action === "complete") {
              resolve(e.data);
            }
          };
          worker.onerror = (error) => reject(error);
          worker.postMessage({ action: "start", P: 10 });
        });

        // console.log("Trace data collected, sending directly to backend...");

        // Create timestamp
        const timestamp = new Date().toISOString();

        // Send trace data directly to backend for processing
        const response = await fetch("/collect_trace", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            trace_data: result.data,
            timestamp: timestamp,
            metadata: result.metadata,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const responseData = await response.json();
        // console.log("Trace data processed:", responseData);
        this.status = "Traced";

        // Add heatmap to local collection
        this.heatmaps.push({
          id: responseData.trace_id,
          image_url: responseData.image_url,
          timestamp: responseData.timestamp,
          stats: responseData.stats,
          predictions: responseData.predictions || [],
          model_used: responseData.model_used || "Unknown Model",
        });

        this.showingTraces = true; // Ensure images are always shown

        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting trace data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Download the trace data as JSON
    async downloadTraces() {
      try {
        this.status = "Downloading traces...";

        const response = await fetch("/api/download_traces");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        // console.log("Downloaded traces:", data);

        // Create download file
        const dataStr = JSON.stringify(data.traces, null, 2);
        const dataBlob = new Blob([dataStr], { type: "application/json" });

        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `cache_traces_${
          new Date().toISOString().split("T")[0]
        }.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        this.status = `Downloaded ${data.traces.length} traces successfully!`;
      } catch (error) {
        console.error("Error downloading traces:", error);
        this.status = `Error downloading traces: ${error.message}`;
        this.statusIsError = true;
      }
    },

    // Refresh results from server
    async refreshResults() {
      try {
        this.status = "Refreshing results...";
        await this.fetchResults();
        this.status = "Results refreshed successfully!";
      } catch (error) {
        console.error("Error refreshing results:", error);
        this.status = `Error refreshing results: ${error.message}`;
        this.statusIsError = true;
      }
    },

    // Clear all results from the server
    async clearResults() {
      try {
        this.status = "Clearing results...";

        const response = await fetch("/api/clear_results", {
          method: "POST",
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        // console.log("Results cleared:", data);

        this.status = "Cleared";

        // Clear local copies
        this.traceData = [];
        this.heatmaps = [];
        this.latencyResults = null;
        this.showingTraces = false;
      } catch (error) {
        console.error("Error clearing results:", error);
        this.status = `Error clearing results: ${error.message}`;
        this.statusIsError = true;
      }
    },

    // Fetch existing results when page loads
    async fetchResults() {
      try {
        const response = await fetch("/api/get_traces");
        if (response.ok) {
          const data = await response.json();
          // console.log("Fetched existing traces:", data);
          this.heatmaps = data.traces.map((trace) => ({
            id: trace.id,
            image_url: trace.image_url,
            timestamp: trace.timestamp,
            stats: this.calculateStats(trace.trace_data),
            predictions: trace.predictions || [],
            model_used: trace.model_used || "Unknown Model",
          }));
          if (this.heatmaps.length > 0) {
            this.showingTraces = true;
          }
        }
      } catch (error) {
        console.error("Error fetching results:", error);
      }
    },

    // Calculate statistics for heatmap display
    calculateStats(traceData) {
      const flatData = traceData.flat();
      const min = Math.min(...flatData);
      const max = Math.max(...flatData);
      // Get dimensions for both original and transposed data
      const rows = traceData.length;
      const cols = traceData[0]?.length || 0;
      return {
        min: min,
        max: max,
        range: max - min,
        samples: rows,
        total_accesses: flatData.reduce((a, b) => a + b, 0),
        shape: [rows, cols],
        transposed_shape: [cols, rows],
      };
    },
  };
}
