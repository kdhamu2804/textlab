import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";

import Layout from "./components/Layout";
import About from "./pages/About";
import Home from "./pages/Home";
import NotFound from "./pages/NotFound";

// The charts library is large, so the Results page is loaded only when opened.
const Results = lazy(() => import("./pages/Results"));

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route
          path="results"
          element={
            <Suspense fallback={<p className="text-ink-3">Loading results…</p>}>
              <Results />
            </Suspense>
          }
        />
        <Route path="about" element={<About />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
