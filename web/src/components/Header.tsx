"use client";

export function Header() {
  return (
    <header className="flex items-center justify-between px-4 py-2.5 border-b border-zinc-200 dark:border-zinc-800">
      <h1 className="text-sm font-bold tracking-tight">
        <span className="text-blue-600">Evidence</span> RAG
      </h1>

      <p className="text-xs text-zinc-500 dark:text-zinc-400">
        Hybrid RRF · NLI · Granite
      </p>
    </header>
  );
}
