# Research notes: RunPod (checked 2026-09-30)

docs.runpod.io, rest.runpod.io and api.runpod.io are blocked from the dev container. The facts below
were read from RunPod's own source repositories:
- `github.com/runpod/docs` @ 07ba10e (2026-09-29), including both OpenAPI specs
- `github.com/runpod/runpodctl` @ 4351fca
- `github.com/runpod/runpod-python` @ 760aea2
- `github.com/runpod/containers` @ 5007471

A file `X.mdx` in the docs repo is the source of `docs.runpod.io/X`. Anything not confirmed is
marked UNVERIFIED. `deploy.py` re-validates against the live API: its first call is a read-only
catalog query, and `--dry-run` prints every payload.

## Decisions derived from these facts
1. **deploy.py targets REST v2** (`https://api.runpod.io/v2`, GA 2026-08-18).
   - REST v1 (`rest.runpod.io/v1`) "is deprecated and will be retired on November 15, 2026", and
     GraphQL is to be retired "early 2027". [docs `api-reference/overview.mdx`, `release-notes.mdx`,
     `api-reference-v2/migrate-from-v1.mdx`]
   - `runpodctl` (v2.14.0, 2026-09-10) still calls v1/GraphQL. The emitted `runpodctl` equivalents are
     therefore best-effort and labelled as such.
2. **Placement loop.**
   - Quote: "This endpoint places one specific GPU type. It does not search for capacity, and it does
     not fall back to a different GPU." [v2 `POST /v2/pods`]
   - deploy.py reads `/v2/catalog/gpus?include=AVAILABILITY&product=POD` and
     `/v2/catalog/datacenters?include=GPU_AVAILABILITY` (filters `regions`, `compliance=GDPR`,
     `networkVolumeTypes`), then tries (GPU type, DC) candidates in order until one returns 201.
   - Retry semantics from the spec:
     - 422: never retry
     - 400: try the next candidate
     - 402: stop (balance)
     - 403: skip
     - 429: back off using `Retry-After`
     - 5xx: retry the same candidate
3. **The network volume pins the DC.**
   - The volume must be "in the same data center as the pod" and is attached only at creation
     (`mounts.network[{volumeId, path}]`, path required, mutually exclusive with `mounts.persistent`).
   - The template therefore carries **no mounts**; the network mount is set on the pod create call.
   - Volume creation is only in DCs whose `networkVolumeTypes` is non-empty. Network volumes are
     Secure Cloud only.
4. **Secrets are created via the API.**
   - `POST /v2/account/secrets {name, value, description?}`; the name must not start with `RUNPOD`.
   - Referenced in env as `{{ RUNPOD_SECRET_<name> }}`, substituted at boot.
   - deploy.py creates/updates `archrender_hf_token` and `archrender_admin_token` from `.env`, so the
     values never appear in template or pod payloads.
   - Docs conflict: `get-started/credentials.mdx` claims names must start with `RUNPOD_SECRET_`. The
     v2 spec is authoritative.
5. **Private GHCR.**
   - `POST /v2/registries {name, username, password}`, referenced as `registry: <id>`.
   - GitHub Packages needs a *classic* PAT with `read:packages`.
   - UNVERIFIED: whether RunPod needs the registry host anywhere. The API has no URL field.
6. **`down` terminates the pod by default** (the network volume survives, "only detached").
   - Restarting a stopped pod can fail with "Zero GPU Pods" because the pod is tied to its host.
     [`pods/troubleshooting/zero-gpus.mdx`]
   - `down --stop` is available when the owner prefers stop. `down --purge` also deletes the volume
     (with confirmation).
7. **OptiX needs `NVIDIA_DRIVER_CAPABILITIES=all` set in the image `ENV`.**
   - `libnvoptix.so`/`libnvidia-rtcore.so` are mounted only with the `graphics` capability.
     [`NVIDIA/libnvidia-container src/nvc_info.c`]
   - `nvidia/cuda` and `runpod/pytorch` images default to `compute,utility`.
   - RunPod's own ComfyUI image sets `ENV NVIDIA_DRIVER_CAPABILITIES=all`.
8. **Upload chunks of 16 MiB stay far below any proxy body cap.** The proxy body limit is UNVERIFIED;
   Cloudflare plans cap at 100 MB.

## API v2 facts used by deploy.py
- **Auth:** `Authorization: Bearer <key>`. Request bodies larger than 102,400 bytes → 413.
- **Errors:** RFC 9457 `application/problem+json` (`title`, `status`, `detail`, `errors[]`).
- **Lists:** `{"<plural>": [...], "pagination": {"nextCursor", "hasNextPage"}}`.
- **Rate limits:** the numbers are UNVERIFIED; read the `RateLimit`/`RateLimit-Policy` headers.
- **Pods.** `POST /v2/pods` fields:
  - `name` (required), `cloud` (`SECURE`|`COMMUNITY`, default SECURE), `templateId`, `image`, `disk`
    (container GB, "ephemeral, wiped on restart"), `env` (map, merged with the template per key),
    `ports`, `registry`, `dataCenterIds` ("preferred"), `mounts`, `startSsh`, `startJupyter`,
    `globalNetworking`, `args`/`cmd`/`entrypoint`.
  - `gpu {id, count, allowedCudaVersions | minCudaVersion (mutually exclusive), minRamPerGpu,
    minVcpuCountPerGpu}`.
  - Not in v2: `interruptible`, `supportPublicIp`, `countryCodes`.
- **Pod status:** `PROVISIONING`, `STARTING`, `RUNNING`, `EXITED`, `ERROR`, `TERMINATED`.
  `runtime.ports` is null until RUNNING. `POST /v2/pods/{id}/action {"action": start|stop|restart|terminate}`
  (409 if invalid). `GET /v2/pods/{id}/logs?source=system|container` (SSE) gives image-pull
  diagnostics. RunPod stops pods on `IMAGE_AUTH_ERROR`, `IMAGE_PULL_ERROR` and `CUDA_VERSION_MISMATCH`.
- **Templates:** `POST /v2/templates` (required `name`, `image`; `disk`, `ports`, `env`, `registry`,
  `allowedCudaVersions`, `category`, `public`, `serverless`). `startSsh`/`startJupyter` **default to
  true**, so set them explicitly. No `readme` field in v2. PATCH to update. A template is a snapshot at
  pod creation.
- **Network volumes:** `POST /v2/network-volumes {name, size (10–4096), dataCenter, type?
  STANDARD|HIGH_PERFORMANCE}`. PATCH can only grow the size.
- **Catalog:** GPUs `{id, name, memory, price{secure, community, serverless}, availability, dataCenters[],
  cudaVersions[]}`. Datacenters `{id, name, region, networkVolumeTypes[], compliance[]}`.

## Proxy, ports, storage
- **Proxy URL** `https://{POD_ID}-{PORT}.proxy.runpod.net`, via Cloudflare with a 100 s max
  connection time → 524. Public, no auth.
  - The docs recommend returning job IDs immediately.
  - Max 10 HTTP ports.
  - WebSocket/SSE idle behaviour is UNVERIFIED, so we keep SSE heartbeats at 15 s and support polling.
- **Storage.**
  - Container disk is lost on stop/restart. The network volume persists independently.
  - Network volume pricing: $0.07/GB/mo < 1 TB, $0.05 beyond. Stopped pod volume disk is $0.20/GB/mo.
  - Throughput is quoted as "200–400 MB/s (up to 10 GB/s peak)". HIGH_PERFORMANCE is "up to 3x
    throughput".
  - Consequence: a 40 GB transformer loads in ~100–200 s from a STANDARD volume. The ModelManager
    keeps evicted weights in pinned CPU RAM where possible. deploy.py prefers HIGH_PERFORMANCE
    volumes where offered.
- **Editing a running pod resets it.** Updating env restarts it.
- **In-pod env:** `RUNPOD_POD_ID`, `RUNPOD_DC_ID`, `RUNPOD_API_KEY` (pod-scoped), `RUNPOD_PUBLIC_IP`,
  `RUNPOD_TCP_PORT_22`, `RUNPOD_VOLUME_ID`, `PUBLIC_KEY`. `runpodctl` is preinstalled in pods. The
  idle watchdog uses `runpodctl pod stop $RUNPOD_POD_ID`, with a REST fallback using a user key
  secret. Pod-scoped key permissions are UNVERIFIED.

## GPU type ids → profiles (ids from the v1 enum + docs table dated 2026-02-12)
- **gpu48:** `NVIDIA L40S`, `NVIDIA RTX 6000 Ada Generation`, `NVIDIA RTX A6000`, `NVIDIA L40`,
  `NVIDIA A40`
- **gpu80:** `NVIDIA H100 80GB HBM3`, `NVIDIA A100-SXM4-80GB`, `NVIDIA A100 80GB PCIe`,
  `NVIDIA H100 PCIe`
- **gpu96plus:** `NVIDIA RTX PRO 6000 Blackwell Server Edition` (96), `NVIDIA H100 NVL` (94, treated
  as gpu96plus with a 94 GB budget), `NVIDIA H200` (141), `NVIDIA H200 NVL` (143), `NVIDIA B200` (180)
- Excluded: AMD, MIG variants.

## Datacenters
- **Network-volume-capable (S3 API list, a subset):** EU-CZ-1, EU-RO-1, EUR-IS-1, EUR-NO-1, US-CA-2,
  US-GA-2, US-IL-1, US-KS-2, US-MD-1, US-MO-1, US-MO-2, US-NC-1, US-NC-2, US-NE-1, US-WA-1.
  EU-SE-1, EU-NL-1 and EU-FR-1 are UNVERIFIED; they are resolved live from the catalog.
- **EU residency candidates:** EU-RO-1 (Romania), EU-CZ-1 (Czechia), EU-SE-1 (Sweden),
  EU-NL-1 (Netherlands), EU-FR-1 (France), EUR-IS-* (Iceland, EEA), EUR-NO-1 (Norway, EEA).

## Official PyTorch template (bootstrap fallback)
- Tag scheme: `runpod/pytorch:<ver>-cu<cuda>-torch<ver>-ubuntu<2204|2404>`. Latest stable:
  `runpod/pytorch:1.3.1-cu1281-torch291-ubuntu2404` (2026-09-15).
- It sets `NVIDIA_REQUIRE_CUDA=cuda>=12.8`, runs `/start.sh` (nginx, optional sshd/jupyter,
  `/pre_start.sh`, `/post_start.sh`), and nginx occupies 8001 (→8000), 9091, 3001, 7861, 8081, 7270.
- **Our app uses port 8000 directly, so our vLLM ports avoid 8001.** vLLM moves to 8101/8102.

## Serverless (documented, not built)
- Handler: `runpod.serverless.start({"handler": fn})`, `progress_update`.
- Network volume at `/runpod-volume`.
- Limits: `/run` 10 MB payload; `/runsync` 20 MB; execution timeout default 600 s (5 s–7 days).
- v2 `CreateEndpointRequest {type QUEUE|LOAD_BALANCER, gpu, templateId, networkVolumes, workers{min,max,idleTimeout}, timeout, flashboot}`.
