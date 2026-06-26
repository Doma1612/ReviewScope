import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { api } from "../api";
import { useTheme } from "../ui/ThemeProvider";

export function AuthPage({ mode }: { mode: "login" | "register" }) {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const { theme, toggleTheme } = useTheme();

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      if (mode === "login") await api.login(email, password);
      else await api.register(email, password);
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    }
  }

  return (
    <main className="auth-page">
      <button className="auth-theme-toggle theme-toggle" onClick={toggleTheme} type="button">
        {theme === "dark" ? "Clear theme" : "Dark theme"}
      </button>
      <form className="auth-card" onSubmit={submit}>
        <h1>{mode === "login" ? "Welcome back" : "Create account"}</h1>
        <p>Analyze document collections with simulated clustering while the app services are built.</p>
        <label>Email<input value={email} onChange={(event) => setEmail(event.target.value)} type="email" required /></label>
        <label>Password<input value={password} onChange={(event) => setPassword(event.target.value)} type="password" required minLength={6} /></label>
        {error && <div className="error">{error}</div>}
        <button className="primary" type="submit">{mode === "login" ? "Login" : "Register"}</button>
        <Link to={mode === "login" ? "/register" : "/login"}>{mode === "login" ? "Need an account?" : "Already have an account?"}</Link>
      </form>
    </main>
  );
}
