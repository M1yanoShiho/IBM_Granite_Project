"use client";

import { useState, useEffect, useCallback } from "react";
import { Settings2 } from "lucide-react";

interface Settings {
  retriever: string;
  selector: string;
  generator: string;
  verifier: string;
}

interface Option {
  id: string;
  label: string;
  desc?: string;
}

interface Available {
  retrievers: Option[];
  selectors: Option[];
  generators: Option[];
  verifiers: Option[];
  current: Settings;
}

interface HeaderProps {
  onSettingsChange: () => void;
}

const SHORT: Record<string, Record<string, string>> = {
  retriever: {
    bm25: "BM25", "strong-bm25": "StBM25", "granite-dense": "Dense",
    "hybrid-rrf": "Hyb-RRF", "hybrid-convex": "Hyb-Cvx",
    query2doc: "Q2Doc", hyde: "HyDE", decompose: "Decomp",
  },
  selector: {
    "top-k": "Top-K", corroboration: "Corrob",
    "gated-corroboration": "Gate", "gated-coverage": "GateCov",
  },
  generator: { local: "Granite", ollama: "Ollama" },
  verifier: {
    off: "V:Off", minicheck: "V:MC", "deberta-base": "V:DB",
    "deberta-large": "V:DL", "true": "V:TRUE", "granite-3b": "V:G3B",
    "granite-8b": "V:G8B",
  },
};

function s(key: string, id: string): string {
  return SHORT[key]?.[id] ?? id.slice(0, 6);
}

export function Header({ onSettingsChange }: HeaderProps) {
  const [config, setConfig] = useState<Available | null>(null);
  const [open, setOpen] = useState(false);

  const fetchConfig = useCallback(async () => {
    try {
      const res = await fetch("/config");
      if (res.ok) setConfig(await res.json());
    } catch { /* offline */ }
  }, []);

  useEffect(() => { fetchConfig(); }, [fetchConfig]);

  async function update(key: string, value: string) {
    if (!config) return;
    const next = { ...config.current, [key]: value };
    await fetch("/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(next),
    });
    await fetchConfig();
    onSettingsChange();
  }

  const cur = config?.current;

  return (
    <header className="flex items-center justify-between px-4 py-2.5 border-b border-zinc-200 dark:border-zinc-800">
      <h1 className="text-sm font-bold tracking-tight">
        <span className="text-blue-600">Evidence</span> RAG
      </h1>

      <div className="relative">
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs
                     bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700
                     text-zinc-600 dark:text-zinc-400 transition-colors"
        >
          <Settings2 size={12} />
          <span className="hidden sm:inline">
            {cur
              ? `${s("retriever", cur.retriever)} · ${s("selector", cur.selector)} · ${s("generator", cur.generator)}`
              : "Settings"}
          </span>
          {cur && cur.verifier !== "off" && (
            <span className="hidden sm:inline text-zinc-400">
              · {s("verifier", cur.verifier)}
            </span>
          )}
        </button>

        {open && (
          <>
            <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
            <div className="absolute right-0 top-full mt-1 z-20 w-72 rounded-lg border border-zinc-200
                            dark:border-zinc-700 bg-white dark:bg-zinc-900 shadow-xl p-4 space-y-4">
              <Group label="Retriever" options={config?.retrievers ?? []} current={cur?.retriever ?? ""}
                onChange={(v) => update("retriever", v)} />
              <Group label="Selector" options={config?.selectors ?? []} current={cur?.selector ?? ""}
                onChange={(v) => update("selector", v)} />
              <Group label="Generator" options={config?.generators ?? []} current={cur?.generator ?? ""}
                onChange={(v) => update("generator", v)} />
              <Group label="Verifier (NLI fact-check)" options={config?.verifiers ?? []} current={cur?.verifier ?? ""}
                onChange={(v) => update("verifier", v)} />
            </div>
          </>
        )}
      </div>
    </header>
  );
}

function Group({
  label, options, current, onChange,
}: {
  label: string;
  options: Option[];
  current: string;
  onChange: (id: string) => void;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-zinc-400 mb-1.5">{label}</div>
      <div className="flex flex-wrap gap-1">
        {options.map((opt) => (
          <button
            key={opt.id}
            onClick={() => onChange(opt.id)}
            title={opt.desc}
            className={`px-2 py-0.5 rounded text-xs transition-colors
              ${opt.id === current
                ? "bg-blue-600 text-white"
                : "bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700"
              }
            `}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
