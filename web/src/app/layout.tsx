import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Evidence RAG Demo",
  description: "Local Perplexity-like demo for the three-module RAG pipeline",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="h-screen bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 antialiased">
        {children}
      </body>
    </html>
  );
}
