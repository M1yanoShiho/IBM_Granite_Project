"use client";

import { Plus, Trash2, MessageSquare, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import type { Conversation } from "@/hooks/useHistory";

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  collapsed: boolean;
  onToggle: () => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  onSwitch: (id: string) => void;
}

export function Sidebar({
  conversations,
  activeId,
  collapsed,
  onToggle,
  onCreate,
  onDelete,
  onSwitch,
}: SidebarProps) {
  if (collapsed) {
    return (
      <button
        onClick={onToggle}
        className="fixed left-0 top-1/2 -translate-y-1/2 z-20 p-2
                   bg-white dark:bg-zinc-900 border border-l-0 border-zinc-200 dark:border-zinc-700
                   rounded-r-lg shadow-lg hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors"
        title="Show history"
      >
        <PanelLeftOpen size={18} className="text-zinc-500" />
      </button>
    );
  }

  return (
    <aside className="w-60 flex-shrink-0 border-r border-zinc-200 dark:border-zinc-800
                      bg-zinc-50 dark:bg-zinc-950 flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3 border-b border-zinc-200 dark:border-zinc-800">
        <button
          onClick={onCreate}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs
                     bg-blue-600 text-white hover:bg-blue-700 transition-colors w-full justify-center"
        >
          <Plus size={14} />
          New Chat
        </button>
        <button
          onClick={onToggle}
          className="ml-1 p-1 rounded hover:bg-zinc-200 dark:hover:bg-zinc-800 flex-shrink-0"
        >
          <PanelLeftClose size={14} className="text-zinc-500" />
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto py-2">
        {conversations.length === 0 ? (
          <p className="px-3 py-4 text-xs text-zinc-400 text-center">
            No conversations yet
          </p>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => onSwitch(conv.id)}
              className={`group flex items-center gap-2 px-3 py-2 cursor-pointer text-xs
                ${conv.id === activeId
                  ? "bg-zinc-200 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100"
                  : "text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800/50"
                }
              `}
            >
              <MessageSquare size={12} className="flex-shrink-0" />
              <span className="truncate flex-1">{conv.title}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(conv.id);
                }}
                className="opacity-0 group-hover:opacity-100 p-0.5 rounded
                           hover:bg-red-100 dark:hover:bg-red-900/30 text-zinc-400 hover:text-red-500
                           transition-all flex-shrink-0"
                title="Delete"
              >
                <Trash2 size={11} />
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
