import { Callout } from "@/components/ui";
import { BackendError } from "@/lib/api";

export function ErrorNotice({ error }: { error: unknown }) {
  if (!error) return null;

  if (error instanceof BackendError) {
    return (
      <Callout tone={error.isUnavailable ? "warning" : "danger"} title={title(error)}>
        <p>{error.message}</p>
        {error.code === "network_error" && (
          <p className="mt-2 font-mono text-2xs">cd backend &amp;&amp; python run.py --warm</p>
        )}
      </Callout>
    );
  }

  return (
    <Callout tone="danger" title="Something went wrong">
      {error instanceof Error ? error.message : String(error)}
    </Callout>
  );
}

function title(error: BackendError): string {
  if (error.code === "network_error") return "Backend unreachable";
  if (error.isUnavailable) return "Component unavailable";
  if (error.code === "invalid_input") return "That study could not be read";
  if (error.code === "payload_too_large") return "File too large";
  return "Request failed";
}
