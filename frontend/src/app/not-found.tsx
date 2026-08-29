import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-2xl px-6 py-16 text-center">
      <h1 className="text-lg font-semibold text-ink">Page not found</h1>
      <p className="mt-2 text-sm text-ink-muted">
        No console exists at this address.
      </p>
      <Link
        href="/"
        className="mt-5 inline-block rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white hover:bg-ink/90"
      >
        Back to the dashboard
      </Link>
    </div>
  );
}
