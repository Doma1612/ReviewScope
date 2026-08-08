import { useSyncExternalStore } from "react";

// Minimal app-wide toast store (no provider needed). Edits call `showToast` to
// surface a transient confirmation with an optional inline "Undo" action; the
// <Toaster /> mounted at the app root renders them and auto-dismisses each.
export type Toast = { id: number; message: string; actionLabel?: string; onAction?: () => void };

let toasts: Toast[] = [];
const listeners = new Set<() => void>();
let nextId = 1;

function emit() {
  for (const listener of listeners) listener();
}
function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function dismissToast(id: number) {
  toasts = toasts.filter((toast) => toast.id !== id);
  emit();
}

export function showToast(toast: Omit<Toast, "id">, timeoutMs = 7000): number {
  const id = nextId++;
  toasts = [...toasts, { ...toast, id }];
  emit();
  if (timeoutMs > 0) setTimeout(() => dismissToast(id), timeoutMs);
  return id;
}

export function Toaster() {
  const items = useSyncExternalStore(subscribe, () => toasts, () => toasts);
  if (!items.length) return null;
  return (
    <div className="toaster" role="status" aria-live="polite">
      {items.map((toast) => (
        <div className="toast" key={toast.id}>
          <span className="toast-message">{toast.message}</span>
          {toast.actionLabel && toast.onAction && (
            <button
              className="toast-action"
              type="button"
              onClick={() => { toast.onAction!(); dismissToast(toast.id); }}
            >
              {toast.actionLabel}
            </button>
          )}
          <button className="toast-close" type="button" aria-label="Dismiss" onClick={() => dismissToast(toast.id)}>×</button>
        </div>
      ))}
    </div>
  );
}
