import type { SSEEvent } from "./types";

// Same-origin when served by FastAPI; override for dev with NEXT_PUBLIC_API_URL
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

/**
 * Calls POST /chat and yields parsed SSE events as they arrive.
 * Throws on network error; server-side errors come as `{event:"error"}`.
 */
export async function* streamChat(query: string): AsyncGenerator<SSEEvent> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`Server returned ${response.status}: ${body}`);
  }

  if (!response.body) {
    throw new Error("No response body — streaming not supported");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    // Keep the last partial line in the buffer
    buffer = lines.pop() ?? "";

    let currentEvent = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const dataStr = line.slice(6);
        if (!dataStr) continue;
        try {
          const data = JSON.parse(dataStr);
          yield { event: currentEvent, data } as SSEEvent;
        } catch {
          // Skip unparseable lines
        }
      }
    }
  }
}

export async function checkHealth(): Promise<{ status: string; pipeline: string }> {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
