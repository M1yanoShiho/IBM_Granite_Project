"use client";

import { useCallback, useRef, useState } from "react";
import { streamChat } from "@/lib/api";
import type { CandidateSet, ChatMessage, SelectionResult } from "@/lib/types";

export type Phase = "idle" | "connecting" | "retrieving" | "selecting" | "generating" | "done";

export interface ChatState {
  messages: ChatMessage[];
  phase: Phase;
  error: string | null;
  /** Full candidate set from the latest retrieval. */
  candidates: CandidateSet | null;
  /** Latest selection result. */
  selection: SelectionResult | null;
}

export function useChat() {
  const [state, setState] = useState<ChatState>({
    messages: [],
    phase: "idle",
    error: null,
    candidates: null,
    selection: null,
  });

  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (userText: string) => {
    // Abort any in-flight request
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const userMsg: ChatMessage = { role: "user", content: userText };
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, userMsg],
      phase: "connecting",
      error: null,
    }));

    try {
      const stream = streamChat(userText);

      // We'll build the assistant message incrementally
      let assistantContent = "";
      let citedIds: string[] | undefined;

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, { role: "assistant", content: "" }],
      }));

      for await (const event of stream) {
        if (controller.signal.aborted) break;

        switch (event.event) {
          case "status":
            setState((prev) => ({
              ...prev,
              phase: event.data.phase as Phase,
            }));
            break;

          case "candidates":
            setState((prev) => ({
              ...prev,
              candidates: event.data,
            }));
            break;

          case "selection":
            setState((prev) => ({
              ...prev,
              selection: event.data,
            }));
            break;

          case "chunk":
            assistantContent += event.data.text;
            setState((prev) => {
              const msgs = [...prev.messages];
              msgs[msgs.length - 1] = {
                ...msgs[msgs.length - 1],
                content: assistantContent,
              };
              return { ...prev, messages: msgs };
            });
            break;

          case "done":
            citedIds = event.data.cited_evidence_ids;
            // If the stream yielded no chunks (e.g. empty answer), set content now
            if (!assistantContent) {
              assistantContent = event.data.answer || "(no answer)";
            }
            setState((prev) => {
              const msgs = [...prev.messages];
              msgs[msgs.length - 1] = {
                ...msgs[msgs.length - 1],
                content: assistantContent,
                citedIds,
              };
              return { ...prev, messages: msgs, phase: "done" };
            });
            break;

          case "error":
            setState((prev) => ({
              ...prev,
              error: event.data.message,
              phase: "done",
            }));
            break;
        }
      }
    } catch (err: unknown) {
      if (controller.signal.aborted) return;
      const message = err instanceof Error ? err.message : "Unknown error";
      setState((prev) => ({ ...prev, error: message, phase: "done" }));
    }
  }, []);

  const stopGenerating = useCallback(() => {
    abortRef.current?.abort();
    setState((prev) => ({ ...prev, phase: "done" }));
  }, []);

  const clearChat = useCallback(() => {
    abortRef.current?.abort();
    setState({
      messages: [],
      phase: "idle",
      error: null,
      candidates: null,
      selection: null,
    });
  }, []);

  const setMessagesFromHistory = useCallback((msgs: ChatMessage[]) => {
    abortRef.current?.abort();
    setState({
      messages: msgs,
      phase: "done",
      error: null,
      candidates: null,
      selection: null,
    });
  }, []);

  return { ...state, sendMessage, stopGenerating, clearChat, setMessagesFromHistory };
}
