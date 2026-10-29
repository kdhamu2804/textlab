import { useCallback, useEffect, useState } from "react";

/**
 * Light/dark theme. Follows the system setting until the user clicks the
 * toggle; their choice is then remembered in localStorage.
 */
export function useTheme(): { dark: boolean; toggle: () => void } {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  const toggle = useCallback(() => {
    setDark((current) => {
      try {
        localStorage.setItem("theme", current ? "light" : "dark");
      } catch {
        /* storage may be blocked; the theme still works for this visit */
      }
      return !current;
    });
  }, []);

  return { dark, toggle };
}

/** Whether the page is currently in dark mode (updates when the theme changes). */
export function useIsDark(): boolean {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  useEffect(() => {
    const observer = new MutationObserver(() =>
      setDark(document.documentElement.classList.contains("dark")),
    );
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);
  return dark;
}

export type AsyncState<T> =
  | { status: "loading" }
  | { status: "error"; error: string }
  | { status: "ready"; data: T };

/** Run an async loader once on mount and track loading / error / data. */
export function useAsync<T>(loader: () => Promise<T>): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading" });
  useEffect(() => {
    let active = true;
    loader()
      .then((data) => active && setState({ status: "ready", data }))
      .catch((e: Error) => active && setState({ status: "error", error: e.message }));
    return () => {
      active = false;
    };
    // Runs once: every caller passes a stable module-level loader.
  }, []);
  return state;
}
