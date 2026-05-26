import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  experimental: {
    useWasmBinary: true,
  },
  turbopack: {
    root: path.resolve(process.cwd()),
  },
};

export default nextConfig;
