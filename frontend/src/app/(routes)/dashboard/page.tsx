"use client";
import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Plus, Clock, LayoutGrid, Server, Activity } from "lucide-react";
import dashboardData from "@/data/dashboard.json";

export default function DashboardPage() {
  const [simulations, setSimulations] = useState<any[]>([]);
  console.log(dashboardData);


  useEffect(() => {
    // In a real app, fetch from API. For now, we simulate fetching the local JSON
    // However, since we are writing to it, we should fetch it via API to see updates
    // But for simplicity in this turn, we can just load the imported JSON or fetch from an API endpoint we created?
    // Let's use the local import for initial render + maybe a fetch
    console.log(dashboardData);

    setSimulations(dashboardData);
  }, []);

  const createSimulation = async () => {
    const name = prompt("Enter simulation name:", "New Simulation");
    if (!name) return;

    try {
      const res = await fetch("/api/simulation/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (data.success) {
        window.location.href = `/dashboard/simulation/${data.id}`;
      }
    } catch (err) {
      console.error(err);
      alert("Failed to create simulation");
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans">
      <header className="mb-12 flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">
            Financial <span className="text-indigo-600">Simulations</span>
          </h1>
          <p className="text-slate-500 mt-2 text-sm max-w-md">
            Manage and run agentic financial network simulations. Connect AI agents to model systemic risk.
          </p>
        </div>
        <button
          onClick={createSimulation}
          className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-xl shadow-lg shadow-indigo-500/20 flex items-center gap-2 font-semibold transition-all active:scale-95"
        >
          <Plus size={18} /> New Simulation
        </button>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {simulations.map((sim) => (
          <Link
            key={sim.id}
            href={`/dashboard/simulation/${sim.id}`}
            className="group relative bg-white rounded-2xl p-6 border border-slate-200 hover:border-indigo-300 hover:shadow-xl transition-all duration-300 flex flex-col justify-between h-64"
          >
            <div className="absolute top-4 right-4 text-slate-300 group-hover:text-indigo-500 transition-colors">
              <Activity size={24} />
            </div>

            <div>
              <div className="w-12 h-12 rounded-xl bg-indigo-50 flex items-center justify-center text-indigo-600 mb-4 group-hover:scale-110 transition-transform">
                <LayoutGrid size={24} />
              </div>
              <h3 className="text-xl font-bold text-slate-800 mb-2">{sim.name}</h3>
              <p className="text-sm text-slate-500 line-clamp-2">
                {sim.description || "Interactive financial network model."}
              </p>
            </div>

            <div className="mt-4 pt-4 border-t border-slate-100 flex items-center gap-4 text-xs text-slate-400 font-medium">
              <div className="flex items-center gap-1">
                <Clock size={14} />
                {new Date(sim.createdAt).toLocaleDateString()}
              </div>
              <div className="flex items-center gap-1">
                <Server size={14} />
                {(Math.random() * 10).toFixed(0)} Nodes
              </div>
            </div>
          </Link>
        ))}

        {/* Placeholder Empty State if needed */}
        {simulations.length === 0 && (
          <div className="col-span-full py-20 text-center text-slate-400 border-2 border-dashed border-slate-200 rounded-3xl">
            <p>No simulations found. Create one to get started.</p>
          </div>
        )}
      </div>
    </div>
  );
}
