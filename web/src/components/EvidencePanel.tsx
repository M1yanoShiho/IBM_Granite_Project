"use client";

import { useEffect, useRef } from "react";
import { PanelRightClose, PanelRightOpen } from "lucide-react";
import type { CandidateSet, SelectionResult } from "@/lib/types";
import { EvidenceCard } from "./EvidenceCard";

interface EvidencePanelProps {
  candidates: CandidateSet | null;
  selection: SelectionResult | null;
  citedIds: string[];
  highlightedId: string | null;
  collapsed: boolean;
  onToggle: () => void;
  onHighlight: (id: string | null) => void;
}

export function EvidencePanel({
  candidates,
  selection,
  citedIds,
  highlightedId,
  collapsed,
  onToggle,
  onHighlight,
}: EvidencePanelProps) {
  const highlightRef = useRef<HTMLDivElement>(null);

  // Scroll highlighted card into view
  useEffect(() => {
    if (highlightedId && highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [highlightedId]);

  if (collapsed) {
    return (
      <button
        onClick={onToggle}
        className="fixed right-0 top-1/2 -translate-y-1/2 z-20 p-2
                   bg-white dark:bg-zinc-900 border border-r-0 border-zinc-200 dark:border-zinc-700
                   rounded-l-lg shadow-lg hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors"
        title="Show evidence panel"
      >
        <PanelRightOpen size={18} className="text-zinc-500" />
      </button>
    );
  }

  const selectedIds = new Set(selection?.items.map((i) => i.evidence_id) ?? []);

  return (
    <aside
      className="w-80 lg:w-96 flex-shrink-0 border-l border-zinc-200 dark:border-zinc-800
                 bg-zinc-50 dark:bg-zinc-950 flex flex-col h-full overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
        <div>
          <h2 className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">
            Evidence
          </h2>
          {candidates && (
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              {candidates.candidates.length} retrieved
              {selection && ` · ${selectedIds.size} selected`}
            </p>
          )}
        </div>
        <button
          onClick={onToggle}
          className="p-1 rounded hover:bg-zinc-200 dark:hover:bg-zinc-800 transition-colors"
          title="Hide evidence panel"
        >
          <PanelRightClose size={16} className="text-zinc-500" />
        </button>
      </div>

      {/* Evidence list */}
      {!candidates || candidates.candidates.length === 0 ? (
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-sm text-zinc-400 dark:text-zinc-500 text-center">
            Evidence will appear here when you ask a question.
          </p>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {candidates.candidates.map((c, i) => {
            const idx = i + 1;
            const isSelected = selectedIds.has(c.evidence_id);
            const isCited = citedIds.includes(c.evidence_id);
            const isHighlighted = highlightedId === c.evidence_id;

            return (
              <div
                key={c.evidence_id}
                ref={isHighlighted ? highlightRef : undefined}
              >
                <EvidenceCard
                  candidate={c}
                  index={idx}
                  isSelected={isSelected}
                  isCited={isCited}
                  highlighted={isHighlighted}
                  onClick={() =>
                    onHighlight(
                      highlightedId === c.evidence_id ? null : c.evidence_id
                    )
                  }
                />
              </div>
            );
          })}
        </div>
      )}
    </aside>
  );
}
