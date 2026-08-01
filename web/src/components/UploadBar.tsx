"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Upload, X, FileText } from "lucide-react";

interface UploadBarProps {
  onUpload: () => void;
}

interface UploadedFile {
  name: string;
}

export function UploadBar({ onUpload }: UploadBarProps) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const fetchFiles = useCallback(async () => {
    try {
      const res = await fetch("/uploads");
      if (res.ok) setFiles(await res.json());
    } catch {
      // server not ready
    }
  }, []);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (![".txt", ".md", ".pdf"].some((ext) => file.name.toLowerCase().endsWith(ext))) {
      alert("Only .txt, .md, and .pdf files are supported.");
      return;
    }

    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/upload", { method: "POST", body: form });
      if (res.ok) {
        await fetchFiles();
        onUpload();
      } else {
        const data = await res.json();
        alert(data.error ?? "Upload failed");
      }
    } catch (err) {
      alert("Upload failed: " + (err instanceof Error ? err.message : "unknown"));
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function handleDelete(name: string) {
    await fetch(`/uploads/${name}`, { method: "DELETE" });
    await fetchFiles();
    onUpload();
  }

  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b border-zinc-200 dark:border-zinc-800 text-xs">
      <input
        ref={inputRef}
        type="file"
        accept=".txt,.md,.pdf"
        onChange={handleUpload}
        className="hidden"
      />
      <button
        onClick={() => inputRef.current?.click()}
        disabled={uploading}
        className="flex items-center gap-1 px-2 py-1 rounded bg-zinc-100 dark:bg-zinc-800
                   hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors
                   text-zinc-600 dark:text-zinc-400 disabled:opacity-50"
      >
        <Upload size={12} />
        {uploading ? "Uploading…" : "Upload"}
      </button>

      {files.map((f) => (
        <span
          key={f.name}
          className="flex items-center gap-1 px-2 py-0.5 rounded bg-blue-50 dark:bg-blue-950/30
                     text-blue-700 dark:text-blue-300"
        >
          <FileText size={10} />
          <span className="max-w-[120px] truncate">{f.name}</span>
          <button
            onClick={() => handleDelete(f.name)}
            className="ml-0.5 hover:text-red-500"
          >
            <X size={10} />
          </button>
        </span>
      ))}
    </div>
  );
}
