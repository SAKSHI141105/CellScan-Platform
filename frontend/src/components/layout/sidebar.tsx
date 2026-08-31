import { NavLink } from "react-router-dom";
import { Activity, FileSpreadsheet, Image as ImageIcon, Info, LayoutDashboard, ScanLine, ScatterChart, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "./theme-toggle";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/clinical-data", label: "Clinical Data", icon: FileSpreadsheet },
  { to: "/upload-image", label: "Upload Image", icon: ImageIcon },
  { to: "/mammography", label: "Mammography", icon: ScanLine },
  { to: "/model-performance", label: "Model Performance", icon: Activity },
  { to: "/cluster-explorer", label: "Cluster Explorer", icon: ScatterChart },
  { to: "/about", label: "About", icon: Info },
];

export function Sidebar({ mobileOpen, onCloseMobile }: { mobileOpen: boolean; onCloseMobile: () => void }) {
  return (
    <>
      {/* backdrop — mobile drawer only, tapping outside closes it */}
      {mobileOpen && (
        <div className="fixed inset-0 z-40 bg-black/50 md:hidden" onClick={onCloseMobile} aria-hidden="true" />
      )}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex h-screen w-64 shrink-0 flex-col bg-sidebar px-3 py-5 text-sidebar-foreground transition-transform duration-200 md:static md:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex items-center gap-2.5 px-2 pb-6">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-accent to-primary text-sm font-bold text-white">
            CS
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-bold leading-tight text-white">CellScan</div>
            <div className="truncate text-[10px] uppercase tracking-wider text-sidebar-foreground/60">
              Decision Support Platform
            </div>
          </div>
          <button onClick={onCloseMobile} className="shrink-0 rounded-md p-1 text-sidebar-foreground/70 hover:text-white md:hidden" aria-label="Close menu">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="mb-4 rounded-lg bg-sidebar-active/40 px-3 py-2.5 text-center text-xs font-semibold text-white">
          CellScan Clinical Suite
        </div>

        <nav className="flex flex-1 flex-col gap-1 overflow-y-auto">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={onCloseMobile}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-sidebar-active text-white"
                    : "text-sidebar-foreground/75 hover:bg-sidebar-active/50 hover:text-white"
                )
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center justify-between border-t border-sidebar-active/60 px-2 pt-4">
          <span className="text-[11px] text-sidebar-foreground/60">Theme</span>
          <ThemeToggle />
        </div>
      </aside>
    </>
  );
}
