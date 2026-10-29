import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { score } from "../format";
import { useIsDark } from "../hooks";

export interface CiValue {
  value: number;
  low: number;
  high: number;
  n: number;
}

export interface Series {
  key: string;
  name: string;
  color: string;
}

interface Props {
  title: string;
  description: string;
  categories: { key: string; name: string }[];
  series: Series[];
  /** values[categoryKey][seriesKey] */
  values: Record<string, Record<string, CiValue | undefined>>;
}

interface TooltipProps {
  active?: boolean;
  label?: string;
  payload?: { dataKey: string; color: string; name: string; payload: Record<string, unknown> }[];
}

function CiTooltip({ active, label, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="card px-3 py-2 text-xs shadow-lg">
      <p className="mb-1 font-medium text-ink">{label}</p>
      {payload.map((p) => {
        const ci = p.payload[`${p.dataKey}_ci`] as CiValue;
        return (
          <p key={p.dataKey} className="flex items-center gap-2 text-ink-2">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: p.color }} />
            {p.name}: <span className="font-medium tabular-nums text-ink">{score(ci.value)}</span>
            <span className="tabular-nums text-ink-3">
              [{score(ci.low)}, {score(ci.high)}] · n={ci.n}
            </span>
          </p>
        );
      })}
    </div>
  );
}

/** Grouped bar chart with 95% CI error bars, a hover tooltip and a table view. */
export default function CiBarChart({ title, description, categories, series, values }: Props) {
  const [showTable, setShowTable] = useState(false);
  const dark = useIsDark();
  const ink = dark ? "#b9c0bb" : "#4d5550";
  const grid = dark ? "#2f3632" : "#e3e1d9";

  const data = categories.map(({ key, name }) => {
    const row: Record<string, unknown> = { name };
    for (const s of series) {
      const ci = values[key]?.[s.key];
      if (!ci) continue;
      row[s.key] = ci.value;
      row[`${s.key}_err`] = [ci.value - ci.low, ci.high - ci.value];
      row[`${s.key}_ci`] = ci;
    }
    return row;
  });

  return (
    <section className="card p-5 sm:p-6">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-semibold">{title}</h3>
          <p className="mt-1 text-sm text-ink-3">{description}</p>
        </div>
        <button
          type="button"
          onClick={() => setShowTable((t) => !t)}
          className="shrink-0 rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-ink-2 hover:bg-surface-2"
        >
          {showTable ? "Show chart" : "Show table"}
        </button>
      </div>

      {showTable ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-ink-3">
                <th className="py-2 pr-4 font-medium" />
                {series.map((s) => (
                  <th key={s.key} className="py-2 pr-4 font-medium">{s.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {categories.map(({ key, name }) => (
                <tr key={key} className="border-b border-border last:border-0">
                  <td className="py-2 pr-4 font-medium">{name}</td>
                  {series.map((s) => {
                    const ci = values[key]?.[s.key];
                    return (
                      <td key={s.key} className="py-2 pr-4 tabular-nums">
                        {ci ? (
                          <>
                            {score(ci.value)}{" "}
                            <span className="text-ink-3">[{score(ci.low)}, {score(ci.high)}]</span>
                          </>
                        ) : (
                          "—"
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="h-72 w-full">
          <ResponsiveContainer>
            <BarChart data={data} margin={{ top: 8, right: 4, left: -16, bottom: 0 }} barCategoryGap="22%" barGap={2}>
              <CartesianGrid vertical={false} stroke={grid} />
              <XAxis dataKey="name" tick={{ fill: ink, fontSize: 12 }} tickLine={false} axisLine={{ stroke: grid }} />
              <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={{ fill: ink, fontSize: 12 }}
                     tickLine={false} axisLine={false} />
              <Tooltip content={<CiTooltip />} cursor={{ fill: grid, opacity: 0.4 }} />
              <Legend
                wrapperStyle={{ fontSize: 12, color: ink, paddingTop: 8 }}
                iconType="square"
                itemSorter={(item) => series.findIndex((s) => s.key === item.dataKey)}
                formatter={(value) => <span style={{ color: ink }}>{value}</span>}
              />
              {series.map((s) => (
                <Bar key={s.key} dataKey={s.key} name={s.name} fill={s.color} radius={[4, 4, 0, 0]}
                     maxBarSize={56} isAnimationActive={false}>
                  <ErrorBar dataKey={`${s.key}_err`} direction="y" width={5} strokeWidth={1.2} stroke={ink}
                            isAnimationActive={false} />
                </Bar>
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}
