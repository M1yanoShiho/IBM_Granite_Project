"use client";

interface StatusBarProps {
  phase: string;
}

const PHASE_LABELS: Record<string, string> = {
  idle: "Ready",
  connecting: "Connecting…",
  retrieving: "Searching evidence…",
  selecting: "Filtering results…",
  generating: "Generating answer…",
  done: "Done",
};

export function StatusBar({ phase }: StatusBarProps) {
  const label = PHASE_LABELS[phase] ?? phase;

  if (phase === "idle" || phase === "done") {
    return (
      <div className="text-xs text-zinc-400 dark:text-zinc-500 px-4 py-1">
        {label}
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-xs text-blue-600 dark:text-blue-400 px-4 py-1">
      <span className="inline-block w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
      {label}
    </div>
  );
}
