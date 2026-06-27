import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, Navigate, Outlet, useNavigate } from "react-router-dom";
import banner from '../images/reviewscope_banner_transparent_dark.svg';

import { api } from "../api";
import { useTheme } from "./ThemeProvider";

export function AppShell() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { theme, toggleTheme } = useTheme();
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, retry: false });

  if (me.isLoading) return <main className="center">Loading ReviewScope...</main>;
  if (me.isError) return <Navigate to="/login" replace />;

  async function logout() {
    await api.logout();
    queryClient.clear();
    navigate("/login");
  }

  return (
    <div>
      <header className="topbar">
        <Link className="brand" to="/"><img src={banner} alt="Logo" /></Link>
        <div className="userbar">
          <Link to="/">Classic</Link>
          <Link to="/deck">Spatial</Link>
          <button className="theme-toggle" onClick={toggleTheme} type="button">
            {theme === "dark" ? "Clear" : "Dark"}
          </button>
          <span>{me.data?.email}</span>
          <button onClick={logout}>Logout</button>
        </div>
      </header>
      <Outlet />
    </div>
  );
}
