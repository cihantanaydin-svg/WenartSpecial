import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative base so the UI works behind https://{POD_ID}-8000.proxy.runpod.net or any prefix (ADR-S12).
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1200 },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
