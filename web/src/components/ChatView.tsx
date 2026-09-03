"use client";

import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage, EvidenceCandidate } from "@/lib/types";
import { StatusBar } from "./StatusBar";

interface ChatViewProps {
  messages: ChatMessage[];
  phase: string;
  error: string | null;
  candidates: EvidenceCandidate[];
  onCiteClick: (evidenceId: string) => void;
}

/**
 * Custom component that renders `[N]` citation references as clickable buttons.
 * The regex matches patterns like [1], [2,3,4], [1,3].
 */
function CitationMark({
  children,
  onCiteClick,
  candidates,
}: {
  children: string;
  onCiteClick: (id: string) => void;
  candidates: EvidenceCandidate[];
}) {
  // Split text on citation patterns and render buttons
  const parts = useMemo(() => {
    const result: Array<{ type: "text" | "cite"; content: string; ids?: string[] }> = [];
    const regex = /\[([\d,\s]+)\]/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(children)) !== null) {
      // Text before this match
      if (match.index > lastIndex) {
        result.push({ type: "text", content: children.slice(lastIndex, match.index) });
      }

      // Parse citation numbers
      const rawNumbers = match[1].split(",").map((s) => parseInt(s.trim(), 10));
      const ids: string[] = [];
      for (const n of rawNumbers) {
        if (n >= 1 && n <= candidates.length) {
          ids.push(candidates[n - 1].evidence_id);
        }
      }
      result.push({ type: "cite", content: match[0], ids });

      lastIndex = match.index + match[0].length;
    }

    // Remaining text
    if (lastIndex < children.length) {
      result.push({ type: "text", content: children.slice(lastIndex) });
    }

    return result;
  }, [children, candidates]);

  return (
    <>
      {parts.map((part, i) =>
        part.type === "cite" && part.ids && part.ids.length > 0 ? (
          <button
            key={i}
            onClick={(e) => {
              e.stopPropagation();
              part.ids!.forEach((id) => onCiteClick(id));
            }}
            className="inline-flex items-center px-1 text-xs font-semibold text-blue-600
                       dark:text-blue-400 bg-blue-50 dark:bg-blue-950/40 rounded
                       hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors
                       cursor-pointer align-middle"
            title={part.ids.join(", ")}
          >
            {part.content}
          </button>
        ) : (
          <span key={i}>{part.content}</span>
        )
      )}
    </>
  );
}

export function ChatView({
  messages,
  phase,
  error,
  candidates,
  onCiteClick,
}: ChatViewProps) {
  return (
    <div className="flex-1 overflow-y-auto">
      <StatusBar phase={phase} />

      {messages.length === 0 ? (
        <div className="flex items-center justify-center h-full px-4 pb-20">
          <div className="text-center max-w-md">
            <h1 className="text-2xl font-bold text-zinc-800 dark:text-zinc-200 mb-2">
              Evidence RAG
            </h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Ask a question. The system retrieves evidence, selects the most
              reliable passages, and generates an answer with citations.
            </p>
          </div>
        </div>
      ) : (
        <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-xl px-4 py-3 text-sm leading-relaxed
                  ${msg.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100"
                  }
                `}
              >
                {msg.role === "assistant" ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        p: ({ children }) => {
                          const str = extractTextContent(children);
                          return (
                            <p className="mb-2 last:mb-0">
                              <CitationMark
                                onCiteClick={onCiteClick}
                                candidates={candidates}
                              >
                                {str}
                              </CitationMark>
                            </p>
                          );
                        },
                        li: ({ children }) => {
                          const str = extractTextContent(children);
                          return (
                            <li>
                              <CitationMark
                                onCiteClick={onCiteClick}
                                candidates={candidates}
                              >
                                {str}
                              </CitationMark>
                            </li>
                          );
                        },
                      }}
                    >
                      {formatAssistantContent(msg.content)}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                )}

                {/* Streaming cursor */}
                {i === messages.length - 1 &&
                  msg.role === "assistant" &&
                  (phase === "generating" || phase === "connecting" || phase === "retrieving" || phase === "selecting") && (
                    <span className="inline-block w-2 h-4 ml-0.5 bg-blue-500 animate-pulse rounded-sm align-middle" />
                  )}
              </div>
            </div>
          ))}

          {error && (
            <div className="max-w-3xl mx-auto px-4">
              <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 text-sm text-red-700 dark:text-red-400">
                {error}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Keep answer text and its evidence declaration on separate display lines. */
function formatAssistantContent(content: string): string {
  const evidenceLabel = /(?:Evidence|证据)\s*[:：]/i.exec(content);
  if (!evidenceLabel || evidenceLabel.index === 0) return content;
  return `${content.slice(0, evidenceLabel.index).trimEnd()}\n\n${content
    .slice(evidenceLabel.index)
    .trimStart()}`;
}

/** Extract plain text from React children for citation matching. */
function extractTextContent(children: React.ReactNode): string {
  if (typeof children === "string") return children;
  if (typeof children === "number") return String(children);
  if (Array.isArray(children)) return children.map(extractTextContent).join("");
  if (children && typeof children === "object" && "props" in children) {
    return extractTextContent((children as { props: { children?: React.ReactNode } }).props.children);
  }
  return "";
}
