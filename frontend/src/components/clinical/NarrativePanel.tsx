import { frame } from "@/components/ui";

export function NarrativePanel({
  text,
  title = "Generated report",
  description,
  bare,
}: {
  text: string;
  title?: string;
  description?: string;
  bare?: boolean;
}) {
  const { Frame, FrameHeader, FrameBody } = frame(bare);

  return (
    <Frame>
      <FrameHeader title={title} description={description} />
      <FrameBody>
        <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-ink-muted">
          {text}
        </pre>
      </FrameBody>
    </Frame>
  );
}
