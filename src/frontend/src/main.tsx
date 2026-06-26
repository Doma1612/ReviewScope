import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "./ui/AppShell";
import { ThemeProvider } from "./ui/ThemeProvider";
import { AuthPage } from "./views/AuthPage";
import { ClusterDetail } from "./views/ClusterDetail";
import { Dashboard } from "./views/Dashboard";
import { DeckDashboard } from "./views/DeckDashboard";
import { PipelineView } from "./views/PipelineView";
import { ProjectView } from "./views/ProjectView";
import { SettingsView } from "./views/SettingsView";
import "./styles.css";

const queryClient = new QueryClient();

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<AuthPage mode="login" />} />
            <Route path="/register" element={<AuthPage mode="register" />} />
            <Route element={<AppShell />}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/deck" element={<DeckDashboard />} />
              <Route path="/projects/:projectId" element={<ProjectView />} />
              <Route path="/projects/:projectId/pipeline" element={<PipelineView />} />
              <Route path="/projects/:projectId/clusters/:clusterId" element={<ClusterDetail />} />
              <Route path="/projects/:projectId/settings" element={<SettingsView />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
