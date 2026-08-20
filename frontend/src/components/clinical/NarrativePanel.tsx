import { Card, CardBody, CardHeader } from "@/components/ui";

export function NarrativePanel({
  text,
  title = "Generated report",
  description,
}: {
  text: string;
  title?: string;
  description?: string;
}) {
  return (
    <Card>
      <CardHeader title={title} description={description} />
      <CardBody>
        <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-ink-muted">
          {text}
        </pre>
      </CardBody>
    </Card>
  );
}
