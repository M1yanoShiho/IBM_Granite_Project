"use client";

import { useState, useEffect, useCallback } from "react";
import type { CandidateSet, ChatMessage, SelectionResult } from "@/lib/types";

export interface EvidenceContext {
  candidates: CandidateSet | null;
  selection: SelectionResult | null;
  citedIds: string[];
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  evidence: EvidenceContext | null;
  createdAt: number;
}

const STORAGE_KEY = "evidence-rag-history";

function load(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function save(convs: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(convs));
}

function newId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

export function useHistory() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  // Load on mount
  useEffect(() => {
    const all = load();
    setConversations(all);
    if (all.length > 0) setActiveId(all[0].id);
  }, []);

  const active = conversations.find((c) => c.id === activeId) ?? null;

  const messages = active?.messages ?? [];
  const title = active?.title ?? "New chat";

  const setMessages = useCallback(
    (msgs: ChatMessage[]) => {
      setConversations((prev) => {
        const next = prev.map((c) =>
          c.id === activeId ? { ...c, messages: msgs } : c
        );
        save(next);
        return next;
      });
    },
    [activeId]
  );

  const createConversation = useCallback(() => {
    const conv: Conversation = {
      id: newId(),
      title: "New chat",
      messages: [],
      evidence: null,
      createdAt: Date.now(),
    };
    setConversations((prev) => {
      const next = [conv, ...prev];
      save(next);
      return next;
    });
    setActiveId(conv.id);
  }, []);

  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      save(next);
      return next;
    });
    setActiveId((prev) => (prev === id ? null : prev));
  }, []);

  const evidence = active?.evidence ?? null;

  const setEvidence = useCallback(
    (ctx: EvidenceContext) => {
      setConversations((prev) => {
        const next = prev.map((c) =>
          c.id === activeId ? { ...c, evidence: ctx } : c
        );
        save(next);
        return next;
      });
    },
    [activeId]
  );

  const updateTitle = useCallback(
    (newTitle: string) => {
      setConversations((prev) => {
        const next = prev.map((c) =>
          c.id === activeId ? { ...c, title: newTitle } : c
        );
        save(next);
        return next;
      });
    },
    [activeId]
  );

  return {
    conversations,
    activeId,
    active,
    messages,
    evidence,
    title,
    setMessages,
    setEvidence,
    createConversation,
    deleteConversation,
    switchTo: setActiveId,
    updateTitle,
  };
}
