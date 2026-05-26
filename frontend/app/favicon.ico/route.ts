export const dynamic = "force-static";

const icon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="7" fill="#2160d8"/>
  <path d="M8 10h16v3H8zm0 6h16v3H8zm0 6h10v3H8z" fill="#fff"/>
</svg>`;

export function GET() {
  return new Response(icon, {
    headers: {
      "content-type": "image/svg+xml",
      "cache-control": "public, max-age=86400",
    },
  });
}
