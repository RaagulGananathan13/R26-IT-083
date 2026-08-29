/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The backend is a separate process on :8000. Proxying through Next means the
  // browser only ever talks to one origin, so no CORS preflight is involved and
  // the API base never has to be baked into the client bundle.
  async rewrites() {
    const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
    return [{ source: "/api/backend/:path*", destination: `${backend}/api/v1/:path*` }];
  },
};

export default nextConfig;
