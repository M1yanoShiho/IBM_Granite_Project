"use client";

import { BookOpen, CheckCircle2, Circle } from "lucide-react";
import type { EvidenceCandidate } from "@/lib/types";

interface EvidenceCardProps {
  candidate: EvidenceCandidate;
  index: number;
  isSelected: boolean;
  isCited: boolean;
  highlighted: boolean;
  onClick: () => void;
}

export function EvidenceCard({
  candidate,
  index,
  isSelected,
  isCited,
  highlighted,
  onClick,
}: EvidenceCardProps) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-3 rounded-lg border transition-all
        ${highlighted
          ? "border-blue-500 bg-blue-50 dark:bg-blue-950/30 ring-2 ring-blue-400"
          : "border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700"
        }
      `}
    >
      <div className="flex items-start gap-2">
        {/* Selection indicator */}
        <span className="flex-shrink-0 mt-0.5">
          {isSelected ? (
            <CheckCircle2 size={14} className="text-green-500" />
          ) : (
            <Circle size={14} className="text-zinc-300 dark:text-zinc-600" />
          )}
        </span>

        <div className="min-w-0 flex-1">
          {/* Header row */}
          <div className="flex items-center gap-2 mb-1">
            <span
              className={`inline-flex items-center justify-center w-5 h-5 rounded text-xs font-mono font-bold
                ${isCited
                  ? "bg-blue-600 text-white"
                  : "bg-zinc-100 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400"
                }
              `}
            >
              {index}
            </span>
            <span className="text-xs text-zinc-500 dark:text-zinc-400 truncate">
              {candidate.source_uri.split("/").pop() ?? candidate.document_id}
            </span>
            <span className="text-xs text-zinc-400 dark:text-zinc-500 ml-auto flex-shrink-0">
              #{candidate.retrieval_rank}
            </span>
          </div>

          {/* Passage text */}
          <p className="text-xs text-zinc-700 dark:text-zinc-300 line-clamp-4 leading-relaxed">
            {candidate.text}
          </p>

          {/* Score */}
          <div className="flex items-center gap-1 mt-1.5">
            <BookOpen size={10} className="text-zinc-400" />
            <span className="text-[10px] text-zinc-400">
              {candidate.retrieval_score.toFixed(4)}
            </span>
          </div>
        </div>
      </div>
    </button>
  );
}
