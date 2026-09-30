// Chunked, resumable uploads (ADR-S08). The upload id is remembered per file so a reload resumes.
import { api, post } from "./api";
import { Sha256 } from "./sha256";

const SLICE = 8 * 1024 * 1024;

async function sha256File(file: File, onProgress: (f: number) => void): Promise<string> {
  const h = new Sha256();
  for (let off = 0; off < file.size; off += SLICE) {
    h.update(new Uint8Array(await file.slice(off, off + SLICE).arrayBuffer()));
    onProgress(Math.min(1, (off + SLICE) / file.size));
  }
  return h.hex();
}

async function sha256Bytes(data: ArrayBuffer): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}

interface UploadInfo { upload_id: string; chunk_size: number; chunks: number }
interface UploadStatus { missing: number[]; chunk_size: number }

export async function uploadFile(
  projectId: string,
  file: File,
  onProgress: (phase: string, fraction: number) => void,
): Promise<string> {
  onProgress("hashing", 0);
  const sha = await sha256File(file, (f) => onProgress("hashing", f));
  const key = `archrender.upload.${projectId}.${sha}`;
  let uploadId = localStorage.getItem(key);
  let chunkSize = 0;
  let missing: number[] = [];
  if (uploadId) {
    try {
      const st = await api<UploadStatus>(`uploads/${uploadId}`);
      chunkSize = st.chunk_size;
      missing = st.missing;
    } catch {
      uploadId = null;
    }
  }
  if (!uploadId) {
    const up = await post<UploadInfo>(`projects/${projectId}/uploads`, { filename: file.name, size: file.size, sha256: sha });
    uploadId = up.upload_id;
    chunkSize = up.chunk_size;
    missing = Array.from({ length: up.chunks }, (_, i) => i);
    localStorage.setItem(key, uploadId);
  }
  const total = Math.ceil(file.size / chunkSize);
  let done = total - missing.length;
  for (const i of missing) {
    const data = await file.slice(i * chunkSize, (i + 1) * chunkSize).arrayBuffer();
    const digest = await sha256Bytes(data);
    for (let attempt = 0; ; attempt++) {
      try {
        await api(`uploads/${uploadId}/chunks/${i}`, { method: "PUT", body: data, headers: { "X-Chunk-SHA256": digest } });
        break;
      } catch (e) {
        if (attempt >= 3) throw e;
        await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
      }
    }
    done += 1;
    onProgress("uploading", done / total);
  }
  const res = await post<{ job_id: string }>(`uploads/${uploadId}/complete`);
  localStorage.removeItem(key);
  return res.job_id;
}
