"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { ChatView } from "@/components/ChatView";
import { ChatInput } from "@/components/ChatInput";
import { EvidencePanel } from "@/components/EvidencePanel";
import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";
import { useChat } from "@/hooks/useChat";
import { useHistory } from "@/hooks/useHistory";

export default function Home() {
  const {
    conversations,
    activeId,
    messages: savedMessages,
    evidence: savedEvidence,
    title: savedTitle,
    setMessages,
    setEvidence,
    createConversation,
    deleteConversation,
    switchTo,
    updateTitle,
  } = useHistory();

  const {
    messages: liveMessages,
    phase,
    error,
    candidates,
    selection,
    sendMessage: sendRaw,
    stopGenerating,
    clearChat,
    setMessagesFromHistory,
  } = useChat();

  const [evidenceCollapsed, setEvidenceCollapsed] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [highlightedId, setHighlightedId] = useState<string | null>(null);

  interface UploadedFile { name: string }
  const [files, setFiles] = useState<UploadedFile[]>([]);

  const fetchFiles = useCallback(async () => {
    try {
      const res = await fetch("/uploads");
      if (res.ok) setFiles(await res.json());
    } catch { /* offline */ }
  }, []);

  useEffect(() => { fetchFiles(); }, [fetchFiles]);

  // Auto-create a conversation if none exists
  const hasCreated = useRef(false);
  useEffect(() => {
    if (!hasCreated.current && conversations.length === 0) {
      hasCreated.current = true;
      createConversation();
    }
  }, [conversations.length, createConversation]);

  // Save evidence context when a query completes
  useEffect(() => {
    if (phase === "done" && candidates) {
      const lastAssistant = [...liveMessages].reverse().find((m) => m.role === "assistant");
      setEvidence({
        candidates,
        selection,
        citedIds: lastAssistant?.citedIds ?? [],
      });
    }
  }, [phase]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync live messages → history
  const lastSave = useRef("");
  useEffect(() => {
    if (!activeId) return;
    const json = JSON.stringify(liveMessages);
    if (json === lastSave.current) return;
    lastSave.current = json;
    setMessages(liveMessages);
    // Auto-title from first user message
    if (savedTitle === "New chat" && liveMessages.length > 0) {
      const first = liveMessages.find((m) => m.role === "user");
      if (first) {
        const t = first.content.slice(0, 40);
        updateTitle(t + (first.content.length > 40 ? "…" : ""));
      }
    }
  }, [liveMessages, activeId, setMessages, savedTitle, updateTitle]);

  // Load history messages when switching conversations
  useEffect(() => {
    if (activeId && savedMessages.length > 0) {
      setMessagesFromHistory(savedMessages);
      lastSave.current = JSON.stringify(savedMessages);
    }
  }, [activeId]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSwitch = useCallback(
    (id: string) => {
      setViewingHistory(true);
      switchTo(id);
      clearChat();
    },
    [switchTo, clearChat]
  );

  const handleNew = useCallback(() => {
    setViewingHistory(false);
    createConversation();
    clearChat();
  }, [createConversation, clearChat]);

  const handleSend = useCallback(
    (text: string) => {
      if (!activeId) return;
      setViewingHistory(false);
      sendRaw(text);
    },
    [activeId, sendRaw]
  );

  const handleDelete = useCallback(
    (id: string) => {
      deleteConversation(id);
      if (id === activeId) clearChat();
    },
    [deleteConversation, activeId, clearChat]
  );

  const handleSettingsChange = useCallback(() => {
    clearChat();
  }, [clearChat]);

  const handleCiteClick = useCallback(
    (evidenceId: string) => {
      if (evidenceCollapsed) setEvidenceCollapsed(false);
      setHighlightedId(evidenceId);
    },
    [evidenceCollapsed]
  );

  const isBusy = !["idle", "done"].includes(phase);
  // Track whether we're viewing live results or history
  const [viewingHistory, setViewingHistory] = useState(false);
  const isLive = !viewingHistory && liveMessages.length > 0 && phase !== "idle";
  const displayCandidates = isLive ? candidates : savedEvidence?.candidates ?? null;
  const displaySelection = isLive ? selection : savedEvidence?.selection ?? null;
  const lastAssistant = [...liveMessages].reverse().find((m) => m.role === "assistant");
  const citedIds = isLive ? (lastAssistant?.citedIds ?? []) : (savedEvidence?.citedIds ?? []);
  const candidateList = displayCandidates?.candidates ?? [];

  return (
    <div className="flex h-full">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((p) => !p)}
        onCreate={handleNew}
        onDelete={handleDelete}
        onSwitch={handleSwitch}
      />

      <main className="flex-1 flex flex-col min-w-0">
        <Header onSettingsChange={handleSettingsChange} />

        <ChatView
          messages={liveMessages}
          phase={phase}
          error={error}
          candidates={candidateList}
          onCiteClick={handleCiteClick}
        />

        <ChatInput
          onSend={handleSend}
          onStop={stopGenerating}
          disabled={isBusy}
          generating={!["idle", "done"].includes(phase)}
          files={files}
          onFilesChanged={fetchFiles}
        />
      </main>

      <EvidencePanel
        candidates={displayCandidates}
        selection={displaySelection}
        citedIds={citedIds}
        highlightedId={highlightedId}
        collapsed={evidenceCollapsed}
        onToggle={() => setEvidenceCollapsed((prev) => !prev)}
        onHighlight={setHighlightedId}
      />
    </div>
  );
}
