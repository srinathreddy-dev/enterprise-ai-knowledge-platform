import AIChat from "./components/ai-chat";
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  FileText,
  FolderOpen,
  LayoutDashboard,
  MessageSquare,
  MoreHorizontal,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Users,
  Zap,
} from "lucide-react";

const documents = [
  {
    name: "Employee Handbook.pdf",
    category: "Human Resources",
    pages: "24 pages",
    updated: "Today",
    color: "blue",
  },
  {
    name: "Leave & Attendance Policy.pdf",
    category: "Company Policies",
    pages: "12 pages",
    updated: "Yesterday",
    color: "violet",
  },
  {
    name: "Employee Benefits.pdf",
    category: "Benefits",
    pages: "18 pages",
    updated: "Nov 8, 2026",
    color: "emerald",
  },
];

const activity = [
  {
    question: "How many vacation days do employees receive?",
    time: "2 minutes ago",
    status: "Answered",
  },
  {
    question: "When does health insurance become active?",
    time: "18 minutes ago",
    status: "Answered",
  },
  {
    question: "What are the retirement plan eligibility rules?",
    time: "42 minutes ago",
    status: "Answered",
  },
];

const bars = [34, 48, 39, 62, 52, 75, 58, 86, 65, 92, 72, 100];

export default function Home() {
  return (
    <main className="min-h-screen bg-[#f7f8fc] text-slate-900">
      <div className="flex min-h-screen">
        {/* Sidebar */}
        <aside className="hidden w-[250px] shrink-0 flex-col bg-[#111827] text-white lg:flex">
          <div className="flex h-[76px] items-center gap-3 border-b border-white/10 px-6">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-violet-600">
              <BrainCircuit size={23} />
            </div>
            <div>
              <p className="text-sm font-bold tracking-wide">NEXUS AI</p>
              <p className="text-[10px] tracking-[2px] text-slate-400">
                KNOWLEDGE PLATFORM
              </p>
            </div>
          </div>

          <div className="px-4 py-7">
            <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[2px] text-slate-500">
              Workspace
            </p>
            <nav className="space-y-1">
              <a href="#overview" className="flex items-center gap-3 rounded-lg bg-blue-600 px-3 py-3 text-sm font-medium text-white">
                <LayoutDashboard size={18} /> Overview
              </a>
              <a href="#knowledge" className="flex items-center gap-3 rounded-lg px-3 py-3 text-sm text-slate-400 transition hover:bg-white/5 hover:text-white">
                <FolderOpen size={18} /> Knowledge Base
              </a>
              <a href="#assistant" className="flex items-center gap-3 rounded-lg px-3 py-3 text-sm text-slate-400 transition hover:bg-white/5 hover:text-white">
                <MessageSquare size={18} /> AI Assistant
              </a>
              <a href="#activity" className="flex items-center gap-3 rounded-lg px-3 py-3 text-sm text-slate-400 transition hover:bg-white/5 hover:text-white">
                <Activity size={18} /> Activity
              </a>
            </nav>

            <p className="mb-3 mt-9 px-3 text-[10px] font-semibold uppercase tracking-[2px] text-slate-500">
              Administration
            </p>
            <nav className="space-y-1">
              <a href="#team" className="flex items-center gap-3 rounded-lg px-3 py-3 text-sm text-slate-400 transition hover:bg-white/5 hover:text-white">
                <Users size={18} /> Team
              </a>
              <a href="#settings" className="flex items-center gap-3 rounded-lg px-3 py-3 text-sm text-slate-400 transition hover:bg-white/5 hover:text-white">
                <Settings size={18} /> Settings
              </a>
              <a href="#help" className="flex items-center gap-3 rounded-lg px-3 py-3 text-sm text-slate-400 transition hover:bg-white/5 hover:text-white">
                <CircleHelp size={18} /> Help & Support
              </a>
            </nav>
          </div>

          <div className="mt-auto p-4">
            <div className="rounded-xl border border-white/10 bg-white/[0.04] p-4">
              <div className="mb-3 flex items-center gap-2 text-sm font-medium">
                <ShieldCheck size={17} className="text-emerald-400" />
                System status
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                Dashboard preview
              </div>
              <p className="mt-2 text-[10px] text-slate-500">
                Live API health is available separately.
              </p>
            </div>
            <div className="mt-5 flex items-center gap-3 border-t border-white/10 pt-5">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-blue-400 to-violet-500 text-xs font-bold">
                SR
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-semibold">Workspace Admin</p>
                <p className="text-[10px] text-slate-500">Administrator</p>
              </div>
              <MoreHorizontal size={18} className="text-slate-500" />
            </div>
          </div>
        </aside>

        {/* Main workspace */}
        <section className="min-w-0 flex-1">
          {/* Top bar */}
          <header className="flex h-[76px] items-center justify-between border-b border-slate-200 bg-white px-5 sm:px-8">
            <div>
              <p className="text-xs text-slate-400">Workspace / Overview</p>
              <p className="text-sm font-semibold text-slate-800">
                Enterprise Knowledge
              </p>
            </div>
            <div className="flex items-center gap-3">
              <div className="hidden items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-xs text-slate-400 sm:flex">
                <Search size={15} />
                Search workspace
                <span className="rounded border border-slate-200 px-1.5 py-0.5 text-[10px]">
                  ⌘ K
                </span>
              </div>
              <button aria-label="Workspace options" className="rounded-lg border border-slate-200 p-2 text-slate-500 hover:bg-slate-50">
                <MoreHorizontal size={19} />
              </button>
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-blue-100 text-xs font-bold text-blue-700">
                SR
              </div>
            </div>
          </header>

          <div className="mx-auto max-w-[1500px] space-y-7 p-5 sm:p-8">
            {/* Welcome */}
            <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
              <div>
                <p className="mb-1 text-xs font-semibold uppercase tracking-[2px] text-blue-600">
                  Enterprise AI workspace
                </p>
                <h1 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
                  Good morning, welcome back.
                </h1>
                <p className="mt-2 text-sm text-slate-500">
                  Your organization&apos;s knowledge, connected and ready to explore.
                </p>
              </div>
              <a href="#assistant" className="inline-flex items-center justify-center gap-2 self-start rounded-lg bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700">
                <Sparkles size={17} />
                Ask AI Assistant
              </a>
            </div>

            {/* KPI cards */}
            <div id="overview" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {[
                {
                  title: "Knowledge documents",
                  value: "03",
                  detail: "Sample documents",
                  icon: FileText,
                  color: "blue",
                },
                {
                  title: "Questions answered",
                  value: "—",
                  detail: "Usage analytics not connected",
                  icon: MessageSquare,
                  color: "violet",
                },
                {
                  title: "Knowledge coverage",
                  value: "—",
                  detail: "Evaluation pending",
                  icon: BrainCircuit,
                  color: "emerald",
                },
                {
                  title: "API protection",
                  value: "Active",
                  detail: "Authentication enabled",
                  icon: ShieldCheck,
                  color: "amber",
                },
              ].map((card) => {
                const Icon = card.icon;
                return (
                  <div key={card.title} className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                    <div className="flex items-center justify-between">
                      <p className="text-xs font-medium text-slate-500">{card.title}</p>
                      <div className={`rounded-lg p-2 ${
                        card.color === "blue" ? "bg-blue-50 text-blue-600" :
                        card.color === "violet" ? "bg-violet-50 text-violet-600" :
                        card.color === "emerald" ? "bg-emerald-50 text-emerald-600" :
                        "bg-amber-50 text-amber-600"
                      }`}>
                        <Icon size={18} />
                      </div>
                    </div>
                    <p className="mt-4 text-2xl font-bold tracking-tight">{card.value}</p>
                    <p className="mt-1 text-[11px] text-slate-400">{card.detail}</p>
                  </div>
                );
              })}
            </div>

            {/* Main grid */}
            <div className="grid gap-6 xl:grid-cols-[1.55fr_1fr]">
              {/* Analytics */}
              <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-bold text-slate-800">Query activity</h2>
                    <p className="mt-1 text-xs text-slate-400">
                      Sample visual — live usage data not connected yet
                    </p>
                  </div>
                  <button className="flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-xs text-slate-600">
                    Last 12 periods <ChevronDown size={14} />
                  </button>
                </div>
                <div className="mt-8 flex h-[190px] items-end gap-2 border-b border-slate-100 pb-2 sm:gap-4">
                  {bars.map((height, index) => (
                    <div key={index} className="flex h-full flex-1 flex-col justify-end">
                      <div
                        className="w-full rounded-t-md bg-gradient-to-t from-blue-600 to-violet-400 opacity-90"
                        style={{ height: `${height}%` }}
                      />
                    </div>
                  ))}
                </div>
                <div className="mt-3 flex justify-between text-[10px] text-slate-400">
                  <span>Period 1</span>
                  <span>Period 4</span>
                  <span>Period 8</span>
                  <span>Period 12</span>
                </div>
              </section>

              {/* Interactive AI assistant */}
              <section
                id="assistant"
                className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"
              >
                <AIChat />
              </section>
            </div>

            {/* Knowledge base and activity */}
            <div className="grid gap-6 xl:grid-cols-[1.55fr_1fr]">
              <section id="knowledge" className="rounded-xl border border-slate-200 bg-white shadow-sm">
                <div className="flex items-center justify-between border-b border-slate-100 p-5">
                  <div>
                    <h2 className="text-sm font-bold">Knowledge Base</h2>
                    <p className="mt-1 text-xs text-slate-400">Your sample knowledge documents</p>
                  </div>
                  <span className="rounded-lg bg-blue-50 px-3 py-2 text-xs font-semibold text-blue-700">
                    3 documents
                  </span>
                </div>
                <div className="divide-y divide-slate-100 px-5">
                  {documents.map((document) => (
                    <div key={document.name} className="flex items-center gap-3 py-4">
                      <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${
                        document.color === "blue" ? "bg-blue-50 text-blue-600" :
                        document.color === "violet" ? "bg-violet-50 text-violet-600" :
                        "bg-emerald-50 text-emerald-600"
                      }`}>
                        <FileText size={19} />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-xs font-semibold text-slate-800">{document.name}</p>
                        <p className="mt-1 text-[10px] text-slate-400">
                          {document.category} · {document.pages}
                        </p>
                      </div>
                      <div className="hidden text-right sm:block">
                        <p className="text-[10px] text-slate-500">{document.updated}</p>
                        <p className="mt-1 flex items-center justify-end gap-1 text-[10px] text-emerald-600">
                          <CheckCircle2 size={12} /> Sample
                        </p>
                      </div>
                      <MoreHorizontal size={17} className="text-slate-400" />
                    </div>
                  ))}
                </div>
                <div className="border-t border-slate-100 p-4">
                  <p className="text-center text-xs text-slate-400">
                    Document upload and management are coming next.
                  </p>
                </div>
              </section>

              <section id="activity" className="rounded-xl border border-slate-200 bg-white shadow-sm">
                <div className="border-b border-slate-100 p-5">
                  <h2 className="text-sm font-bold">Recent activity</h2>
                  <p className="mt-1 text-xs text-slate-400">Example questions from our API tests</p>
                </div>
                <div className="divide-y divide-slate-100 px-5">
                  {activity.map((item) => (
                    <div key={item.question} className="py-4">
                      <p className="text-xs font-medium leading-5 text-slate-700">{item.question}</p>
                      <div className="mt-2 flex items-center justify-between">
                        <span className="flex items-center gap-1 text-[10px] text-slate-400">
                          <Clock3 size={12} /> {item.time}
                        </span>
                        <span className="rounded-full bg-emerald-50 px-2 py-1 text-[10px] font-medium text-emerald-700">
                          {item.status}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </div>

            {/* Footer */}
            <footer className="flex flex-col justify-between gap-2 border-t border-slate-200 py-5 text-[10px] text-slate-400 sm:flex-row">
              <p>© 2026 Nexus AI · Enterprise Knowledge Platform</p>
              <p className="flex items-center gap-2">
                <Zap size={12} /> Dashboard prototype · Analytics are illustrative
              </p>
            </footer>
          </div>
        </section>
      </div>
    </main>
  );
}
