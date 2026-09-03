"use client";

import { useState, useRef, type FormEvent, type KeyboardEvent } from "react";
import { Send, Paperclip, X, FileText, Square } from "lucide-react";

interface UploadedFile {
  name: string;
}

interface ChatInputProps {
  onSend: (text: string) => void;
  onStop: () => void;
  disabled: boolean;
  generating: boolean;
  files: UploadedFile[];
  onFilesChanged: () => void;
}

export function ChatInput({ onSend, onStop, disabled, generating, files, onFilesChanged }: ChatInputProps) {
  const [text, setText] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function submit() {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    submit();
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/upload", { method: "POST", body: form });
      if (!res.ok) {
        const data = await res.json();
        alert(data.error ?? "Upload failed");
      } else {
        onFilesChanged();
      }
    } catch (err) {
      alert("Upload failed");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleDelete(name: string) {
    await fetch(`/uploads/${name}`, { method: "DELETE" });
    onFilesChanged();
  }

  const isBusy = disabled || uploading;

  return (
    <form
      onSubmit={handleSubmit}
      className="border-t border-zinc-200 dark:border-zinc-800 p-4"
    >
      {/* Uploaded files tags */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-2 max-w-3xl mx-auto">
          {files.map((f) => (
            <span
              key={f.name}
              className="flex items-center gap-1 px-2 py-0.5 rounded text-xs
                         bg-blue-50 dark:bg-blue-950/30 text-blue-700 dark:text-blue-300"
            >
              <FileText size={10} />
              <span className="max-w-[150px] truncate">{f.name}</span>
              <button type="button" onClick={() => handleDelete(f.name)} className="hover:text-red-500">
                <X size={10} />
              </button>
            </span>
          ))}
        </div>
      )}

      <div className="flex items-end gap-2 max-w-3xl mx-auto">
        {/* Upload button */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.md,.pdf"
          onChange={handleFile}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isBusy}
          className="flex-shrink-0 p-2.5 rounded-lg text-zinc-400 hover:text-zinc-600
                     dark:hover:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800
                     disabled:opacity-40 transition-colors"
          title="Upload file"
        >
          <Paperclip size={18} />
        </button>

        {/* Text input */}
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question…"
          rows={1}
          disabled={isBusy}
          className="flex-1 resize-none rounded-lg border border-zinc-300 dark:border-zinc-700
                     bg-white dark:bg-zinc-900 px-4 py-3 text-sm
                     text-zinc-900 dark:text-zinc-100
                     placeholder:text-zinc-400 dark:placeholder:text-zinc-500
                     focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                     disabled:opacity-50"
        />

        {/* Send / Stop button */}
        {generating ? (
          <button
            type="button"
            onClick={onStop}
            className="flex-shrink-0 p-3 rounded-lg bg-red-600 text-white
                       hover:bg-red-700 transition-colors"
            title="Stop generating"
          >
            <Square size={14} fill="currentColor" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={isBusy || !text.trim()}
            className="flex-shrink-0 p-3 rounded-lg bg-blue-600 text-white
                       hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed
                       transition-colors"
          >
            <Send size={16} />
          </button>
        )}
      </div>
    </form>
  );
}
