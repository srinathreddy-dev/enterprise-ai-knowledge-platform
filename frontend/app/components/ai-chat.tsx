"use client";

import { FormEvent, useState } from "react";
import {
  Bot,
  FileText,
  Loader2,
  Send,
  Sparkles,
  User,
} from "lucide-react";

type Evidence = {
  document_name: string;
  page_number: number;
  chunk_id: string;
  text: string;
  relevance_score: number;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  evidence?: Evidence[];
};

export default function AIChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I'm your Enterprise AI Assistant. Ask me a question about your organization's knowledge base.",
    },
  ]);

  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || loading) return;

    if (trimmedQuestion.length > 2000) {
      setError("Your question must be 2,000 characters or fewer.");
      return;
    }

    setError("");
    setQuestion("");

    setMessages((previous) => [
      ...previous,
      { role: "user", content: trimmedQuestion },
    ]);

    setLoading(true);

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmedQuestion }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || `Request failed (${response.status}).`
        );
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
          evidence: data.evidence || [],
        },
      ]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to contact the AI service."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="flex min-h-[520px] flex-col">
      <div className="flex items-center gap-3 border-b border-slate-100 p-5">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 text-white">
          <Bot size={21} />
        </div>

        <div>
          <h2 className="text-sm font-bold">AI Assistant</h2>
          <p className="text-[11px] text-slate-400">
            Powered by your enterprise knowledge base
          </p>
        </div>

        <span className="ml-auto flex items-center gap-1.5 text-[10px] font-medium text-emerald-600">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
          Connected
        </span>
      </div>

      <div className="flex-1 space-y-5 overflow-y-auto p-5">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex gap-3 ${
              message.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            {message.role === "assistant" && (
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
                <Bot size={17} />
              </div>
            )}

            <div
              className={`max-w-[85%] space-y-3 ${
                message.role === "user" ? "order-first" : ""
              }`}
            >
              <div
                className={`rounded-xl p-3 text-xs leading-6 ${
                  message.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-slate-100 text-slate-700"
                }`}
              >
                {message.content}
              </div>

              {message.evidence && message.evidence.length > 0 && (
                <div className="space-y-2">
                  <details className="group">
                    <summary className="flex cursor-pointer list-none items-center gap-2 rounded-lg bg-blue-50 px-3 py-3 text-[11px] font-semibold text-blue-700 hover:bg-blue-100">
                      <Sparkles size={14} />
                      Supporting evidence ({Math.min(message.evidence.length, 3)})
                      <span className="ml-auto text-[10px] font-normal text-blue-500 group-open:hidden">
                        Show sources +
                      </span>
                      <span className="ml-auto hidden text-[10px] font-normal text-blue-500 group-open:inline">
                        Hide sources −
                      </span>
                    </summary>

                    <div className="mt-3 space-y-2">
                      {message.evidence.slice(0, 3).map((item, evidenceIndex) => (
                        <div
                          key={`${item.chunk_id}-${evidenceIndex}`}
                          className="rounded-lg border border-blue-100 bg-blue-50/60 p-3"
                        >
                          <div className="flex items-center gap-2 text-[10px] font-semibold text-blue-800">
                            <FileText size={13} />
                            {item.document_name}
                            <span className="ml-auto">
                              Page {item.page_number}
                            </span>
                          </div>

                          <p className="mt-2 text-xs leading-5 text-slate-600">
                            {item.text}
                          </p>

                          <p className="mt-2 text-[10px] text-slate-400">
                            Relevance score:{" "}
                            {Number(item.relevance_score).toFixed(3)}
                          </p>
                        </div>
                      ))}
                    </div>
                  </details>
                </div>
              )}
            </div>

            {message.role === "user" && (
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-200 text-slate-600">
                <User size={16} />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <Loader2 size={17} className="animate-spin text-blue-600" />
            Searching documents and generating your answer...
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700"
          >
            {error}
          </div>
        )}
      </div>

      <form
        onSubmit={handleSubmit}
        className="border-t border-slate-100 p-4"
      >
        <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white p-2 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-50">
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask your organization anything..."
            maxLength={2000}
            disabled={loading}
            aria-label="Ask a question"
            className="min-w-0 flex-1 bg-transparent px-2 py-2 text-xs text-slate-700 outline-none placeholder:text-slate-400 disabled:opacity-50"
          />

          <button
            type="submit"
            disabled={loading || !question.trim()}
            aria-label="Send question"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Send size={16} />
            )}
          </button>
        </div>

        <p className="mt-2 text-center text-[10px] text-slate-400">
          Answers are generated from retrieved knowledge base evidence.
        </p>
      </form>
    </section>
  );
}
