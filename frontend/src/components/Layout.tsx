import { Moon, Sun } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

import { useTheme } from "../hooks";
import CrisisNote from "./CrisisNote";

const NAV = [
  { to: "/", label: "Analyze" },
  { to: "/results", label: "Results" },
  { to: "/about", label: "About" },
];

function Logo() {
  return (
    <NavLink to="/" className="flex items-center gap-2.5 font-semibold tracking-tight">
      <img src="/favicon.svg" alt="" className="h-8 w-8" />
      <span className="hidden text-lg sm:inline">TextLab</span>
    </NavLink>
  );
}

export default function Layout() {
  const { dark, toggle } = useTheme();

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b border-border bg-bg/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-5xl items-center justify-between gap-3 px-4">
          <Logo />
          <nav className="flex items-center gap-1">
            {NAV.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end
                className={({ isActive }) =>
                  `rounded-lg px-2.5 py-2 text-sm font-medium transition-colors sm:px-3 ${
                    isActive ? "bg-accent-soft text-accent" : "text-ink-2 hover:bg-surface-2 hover:text-ink"
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
            <button
              type="button"
              onClick={toggle}
              aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
              className="ml-1 rounded-lg p-2 text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink"
            >
              {dark ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-10">
        <Outlet />
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto flex max-w-5xl flex-col gap-3 px-4 py-6 text-sm text-ink-3 sm:flex-row sm:items-center sm:justify-between">
          <CrisisNote compact />
          <p>TextLab · Saint Louis University · MS in AI</p>
        </div>
      </footer>
    </div>
  );
}
