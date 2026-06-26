import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FormEvent, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { api } from "../api";

export function SettingsView() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api.project(projectId) });
  const members = useQuery({ queryKey: ["members", projectId], queryFn: () => api.members(projectId) });
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const rename = useMutation({ mutationFn: () => api.updateProject(projectId, name || project.data?.name || "Untitled"), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["project", projectId] }) });
  const invite = useMutation({ mutationFn: () => api.addMember(projectId, email), onSuccess: () => { setEmail(""); queryClient.invalidateQueries({ queryKey: ["members", projectId] }); } });
  const remove = useMutation({ mutationFn: (userId: string) => api.removeMember(projectId, userId), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["members", projectId] }) });
  const deleteProject = useMutation({ mutationFn: () => api.deleteProject(projectId), onSuccess: () => navigate("/") });

  const owner = project.data?.role === "owner";

  function submitRename(event: FormEvent) { event.preventDefault(); rename.mutate(); }
  function submitInvite(event: FormEvent) { event.preventDefault(); invite.mutate(); }

  return (
    <main className="page">
      <Link to={project.data?.status === "ready" ? `/projects/${projectId}` : "/"}>Back</Link>
      <section className="page-header"><div><h1>Project Settings</h1><p>{project.data?.name}</p></div></section>
      <section className="card"><h2>Rename</h2><form className="inline-form" onSubmit={submitRename}><input disabled={!owner} placeholder={project.data?.name} value={name} onChange={(event) => setName(event.target.value)} /><button disabled={!owner}>Save</button></form></section>
      <section className="card"><h2>Sharing</h2>{owner && <form className="inline-form" onSubmit={submitInvite}><input type="email" placeholder="registered@email.com" value={email} onChange={(event) => setEmail(event.target.value)} /><button>Invite viewer</button></form>}{invite.error && <div className="error small">{invite.error.message}</div>}{members.data?.map((member) => <div className="member" key={member.user_id}><span>{member.email}</span><strong>{member.role}</strong>{owner && member.role !== "owner" && <button onClick={() => remove.mutate(member.user_id)}>Revoke</button>}</div>)}</section>
      {owner && <section className="card danger"><h2>Danger zone</h2><button onClick={() => confirm("Delete this project?") && deleteProject.mutate()}>Delete project</button></section>}
    </main>
  );
}
