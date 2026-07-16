"""
===================================================================================
PHASE 9 — CLOUD & DEVOPS (HashedIn by Deloitte — Lead Python/GenAI Expert)
===================================================================================

WHY THIS PHASE:
    The JD: "Strong understanding on at-least one cloud platform (AWS, GCP,
    Azure) to deploy, manage, and scale applications" + "Strong proficiency
    with Git... collaborative workflows (branching, pull/merge requests, code
    reviews)" + "translate POCs into well-architected, scalable, production-
    ready solutions." A lead owns how the system SHIPS and RUNS, not just the code.

DEPTH LEVEL: Technical Lead. Enough cloud/DevOps to deploy, scale, and operate
    an AI service and to lead the team's delivery workflow. (You don't need to
    be a DevOps engineer — you need to architect and direct deployment.)

SECTIONS:
    1.  Containers (Docker) — Package Once, Run Anywhere
    2.  Orchestration (Kubernetes) — Run at Scale
    3.  The Cloud Landscape (AWS focus) — Core Services for AI Apps
    4.  Deploying a Python/AI Service (the real path)
    5.  CI/CD Pipelines
    6.  Infrastructure as Code (IaC)
    7.  Git & Collaborative Workflows (the JD names this)
    8.  Monitoring, Logging & Operations (Day-2)
    9.  Cost, Security & Scaling in the Cloud
    10. Deploying LLM/AI Workloads Specifically
    11. Interview Q&A (lead-level)
    12. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: CONTAINERS (Docker) — Package Once, Run Anywhere
# =================================================================================
'''
Containers are the foundation of modern deployment. A lead must know the why
and the craft of a good image.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT A CONTAINER IS:
    A lightweight, isolated package of your app + its dependencies + runtime,
    that runs identically on any machine with a container runtime. Solves
    "works on my machine" — the same image runs in dev, CI, and prod.

CONTAINER vs VM (the classic question):
    VM: virtualizes HARDWARE; each VM has a full guest OS. Heavy (GBs), slow to
        boot (minutes), strong isolation.
    CONTAINER: virtualizes the OS; shares the host KERNEL, isolates the process.
        Light (MBs), fast to start (seconds), runs many per host.
    -> Containers are the unit of deployment; you run many containers per VM.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A PRODUCTION-GRADE DOCKERFILE (FastAPI example) + the lead's best practices:

    # Use a slim, pinned base (small + reproducible)
    FROM python:3.11-slim AS base
    ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

    WORKDIR /app
    # Copy deps FIRST + install -> Docker layer caching (deps don't change
    # every code edit, so this layer is cached -> fast rebuilds)
    COPY pyproject.toml uv.lock ./
    RUN pip install uv && uv sync --frozen --no-dev

    # Then copy code (changes often -> later layer)
    COPY src/ ./src/

    # Run as a NON-ROOT user (security)
    RUN useradd -m appuser
    USER appuser

    EXPOSE 8000
    CMD ["uvicorn", "myapp.main:app", "--host", "0.0.0.0", "--port", "8000"]

    KEY PRACTICES (each is a talking point):
    - LAYER ORDER for caching: copy deps + install before copying code, so a
      code change doesn't reinstall all dependencies.
    - MULTI-STAGE builds: a 'builder' stage compiles/installs, the final stage
      copies only what's needed -> small, clean image.
    - SLIM/pinned base image (python:3.11-slim) -> smaller + reproducible +
      smaller attack surface.
    - NON-ROOT user -> security.
    - .dockerignore (exclude .venv, .git, tests) -> smaller, faster builds.
    - Pin versions (lock file + --frozen) -> reproducible images.

INTERVIEW ANSWER:
    "A container packages the app, its dependencies, and runtime so it runs
    identically everywhere — solving 'works on my machine.' Unlike a VM, which
    virtualizes hardware with a full guest OS, a container shares the host
    kernel and isolates the process, so it's megabytes and starts in seconds; I
    run many per host. For the Dockerfile I order layers for caching — copy the
    lock file and install dependencies before copying code, so a code change
    doesn't reinstall everything — use a slim pinned base and multi-stage builds
    for a small image, run as a non-root user, and use a .dockerignore. Pinning
    via the lock file with frozen installs keeps images reproducible."
'''


# =================================================================================
# SECTION 2: ORCHESTRATION (Kubernetes) — Run at Scale
# =================================================================================
'''
You don't need to be a K8s expert, but a lead must understand it enough to
architect deployment + scaling. Know the concepts and WHY.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY ORCHESTRATION:
    Running one container is easy. Running hundreds across machines — with
    scaling, self-healing, rolling updates, networking, secrets — needs an
    orchestrator. Kubernetes is the standard.

CORE K8S CONCEPTS (the vocabulary to speak fluently):
    - POD: the smallest unit — one (or a few tightly-coupled) container(s).
      Ephemeral; can die and be replaced.
    - DEPLOYMENT: declares the DESIRED state (e.g., "5 replicas of this image");
      K8s reconciles reality to match. Handles rolling updates + rollback.
    - SERVICE: a stable network endpoint + load balancing across pods (pods come
      and go; the Service IP is stable).
    - INGRESS: routes external HTTP(S) traffic to Services (the front door + TLS).
    - CONFIGMAP / SECRET: inject config / secrets into pods (env or files).
    - HPA (Horizontal Pod Autoscaler): auto-scale replicas on CPU / memory /
      custom metrics (e.g., request latency, queue depth).
    - NAMESPACE: logical isolation (dev/staging/prod or per-team).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE TWO HEALTH PROBES (a guaranteed question):
    - LIVENESS probe: "is the app alive?" If it fails, K8s RESTARTS the pod.
    - READINESS probe: "is the app ready for traffic?" If it fails, K8s stops
      routing to it (but doesn't restart). Used during startup / warmup / when
      a dependency is down.
    For FastAPI: a /health endpoint backs both. Critical for zero-downtime.

THE DECLARATIVE MODEL (the K8s philosophy):
    You declare the DESIRED state (YAML manifests); the control plane
    continuously RECONCILES actual -> desired. Self-healing (crashed pod
    replaced), self-scaling (HPA). You describe WHAT, not HOW.

DO YOU ALWAYS NEED K8S? (the senior nuance):
    No. K8s is powerful but complex. For simpler needs: managed container
    services (AWS ECS/Fargate, Google Cloud Run, Azure Container Apps) give
    autoscaling containers without K8s overhead. "Use the least operational
    complexity that meets the scale/control needs."

INTERVIEW ANSWER:
    "Kubernetes orchestrates containers at scale — scaling, self-healing,
    rolling updates, networking. The core objects: a Pod wraps the container; a
    Deployment declares desired replicas and handles rolling updates and
    rollback; a Service is a stable load-balanced endpoint over ephemeral pods;
    Ingress routes external traffic with TLS; ConfigMaps and Secrets inject
    config; and the HPA autoscales replicas on CPU or custom metrics like
    latency. I always set liveness and readiness probes against a /health
    endpoint — liveness restarts a dead pod, readiness stops routing to one
    that isn't ready — which is what enables zero-downtime deploys. The model is
    declarative: you declare desired state and the control plane reconciles to
    it. That said, K8s is complex, so for simpler workloads I'd use managed
    container services like ECS Fargate or Cloud Run — least operational
    complexity that meets the need."
'''


# =================================================================================
# SECTION 3: THE CLOUD LANDSCAPE (AWS focus) — Core Services for AI Apps
# =================================================================================
'''
The JD wants "at least one cloud." Go deep on ONE (AWS here) + know the GCP/
Azure equivalents. Map services to what an AI app needs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLOUD SERVICE CATEGORIES + AWS / GCP / AZURE EQUIVALENTS:

    NEED                  AWS                  GCP                AZURE
    Compute (VMs)         EC2                  Compute Engine     VMs
    Containers (managed)  ECS / Fargate / EKS  Cloud Run / GKE    Container Apps/AKS
    Serverless functions  Lambda               Cloud Functions    Functions
    Object storage        S3                   Cloud Storage      Blob Storage
    Relational DB         RDS / Aurora         Cloud SQL          Azure SQL
    NoSQL                 DynamoDB             Firestore          Cosmos DB
    Cache                 ElastiCache (Redis)  Memorystore        Cache for Redis
    Queue / messaging     SQS / SNS / Kinesis  Pub/Sub            Service Bus
    API Gateway           API Gateway          API Gateway/Apigee API Management
    Secrets               Secrets Manager      Secret Manager     Key Vault
    Managed LLMs          Bedrock              Vertex AI          Azure OpenAI
    Vector search         OpenSearch/Aurora pgv Vertex Vector     AI Search
    Load balancer         ELB/ALB              Cloud Load Bal.    Load Balancer
    Monitoring            CloudWatch           Cloud Monitoring   Monitor

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A TYPICAL AI APP ON AWS (mapping the Phase 7 architecture to services):
    - Client -> CloudFront (CDN) + Route53 (DNS)
    - ALB (load balancer) -> ECS Fargate / EKS running the FastAPI containers
    - Bedrock (or Azure OpenAI) for LLMs via the LLM gateway
    - RDS Postgres (with pgvector) for data + embeddings; ElastiCache (Redis)
      for caching
    - S3 for raw documents; SQS + Lambda/workers for async ingestion
    - Secrets Manager for keys; CloudWatch + X-Ray for observability
    - IAM for least-privilege access between services

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE SHARED RESPONSIBILITY MODEL (cloud security basics):
    Cloud provider secures the cloud (hardware, infra); YOU secure what's IN it
    (your app, data, access config, IAM). Most breaches are misconfiguration on
    the customer side (open S3 buckets, over-broad IAM).

IAM — THE CORE CLOUD SECURITY CONCEPT:
    Identity & Access Management — WHO can do WHAT on WHICH resource. Principle
    of LEAST PRIVILEGE: grant the minimum permissions needed. Use roles, not
    long-lived keys, for service-to-service.

INTERVIEW ANSWER:
    "I'm strongest on AWS. I map the architecture to services: ALB in front of
    FastAPI containers on ECS Fargate or EKS, Bedrock or Azure OpenAI for the
    LLMs behind a gateway, RDS Postgres with pgvector for data and embeddings,
    ElastiCache Redis for caching, S3 for documents, SQS plus Lambda or workers
    for async ingestion, Secrets Manager for keys, and CloudWatch with X-Ray for
    observability. I know the GCP and Azure equivalents — Cloud Run or GKE,
    Vertex AI, Cloud SQL. Two fundamentals I always apply: the shared
    responsibility model — the provider secures the cloud, I secure what's in
    it — and IAM with least privilege using roles rather than long-lived keys."
'''


# =================================================================================
# SECTION 4: DEPLOYING A PYTHON/AI SERVICE (the real path)
# =================================================================================
'''
"How do you deploy your app?" — walk the full path confidently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DEPLOYMENT PIPELINE (code -> running in prod):
    1. CODE merged to main (after PR + CI passes).
    2. BUILD a Docker image; tag it (git SHA + semver).
    3. PUSH to a container registry (ECR / GCR / ACR / Docker Hub).
    4. DEPLOY: update the K8s Deployment / ECS service to the new image tag.
    5. ROLLING UPDATE: new pods come up, pass readiness, old pods drain -> zero
       downtime. Roll back instantly if health fails.
    6. VERIFY: smoke tests + monitoring + alerts.

DEPLOYMENT STRATEGIES (know the trade-offs):
    - ROLLING UPDATE (default): gradually replace old pods with new. Zero
      downtime, simple; brief mixed-version window.
    - BLUE-GREEN: run new version (green) alongside old (blue); switch traffic
      all at once; instant rollback by switching back. Needs 2x resources.
    - CANARY: route a small % to the new version, watch metrics, ramp up if
      healthy, roll back if not. Safest for risky changes; more complex.
    - FEATURE FLAGS: decouple deploy from release — ship code dark, enable per
      segment. (Ties to your prompt-versioning lesson.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ZERO-DOWNTIME REQUIREMENTS:
    - Readiness probes (don't route until ready).
    - Graceful shutdown: catch SIGTERM, stop taking new requests, finish
      in-flight ones, then exit (FastAPI lifespan / signal handling).
    - Backward-compatible DB migrations (expand-then-contract — Phase 4): never
      a migration that breaks the currently-running old version.

INTERVIEW ANSWER:
    "The pipeline: merge to main after CI passes, build a Docker image tagged
    with the git SHA, push to a registry like ECR, then update the Deployment
    to the new tag. Kubernetes does a rolling update — new pods come up, pass
    readiness, old pods drain — for zero downtime, with instant rollback if
    health fails. For risky changes I use canary, routing a small percentage
    first and watching metrics, or blue-green for instant switch-back. Zero
    downtime needs readiness probes, graceful shutdown on SIGTERM to finish
    in-flight requests, and backward-compatible migrations using
    expand-then-contract so the change never breaks the running old version. I
    also separate deploy from release with feature flags."
'''


# =================================================================================
# SECTION 5: CI/CD PIPELINES
# =================================================================================
'''
The JD: translate POCs to production; a lead owns the delivery pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CI (Continuous Integration) — on every PR/push:
    1. Checkout + set up Python (+ cache deps).
    2. LINT + FORMAT check (ruff/black) + TYPE check (mypy).
    3. RUN TESTS (pytest) + coverage gate.
    4. (AI) optional eval suite on prompt/model changes.
    5. BUILD the Docker image (and scan it for vulnerabilities).
    -> Fast feedback; nothing merges that breaks the build/tests.

CD (Continuous Delivery/Deployment):
    - DELIVERY: every passing main build is deployable; deploy is a manual
      approval (common for prod).
    - DEPLOYMENT: auto-deploy to prod on green (mature teams, strong tests).
    Typical: auto-deploy to dev/staging, manual approval gate to prod.

TOOLS: GitHub Actions, GitLab CI, Jenkins, CircleCI, Azure DevOps.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A GITHUB ACTIONS PIPELINE (conceptual):
    on: [pull_request, push to main]
    jobs:
      test:   ruff check . ; mypy . ; pytest --cov (fail under 80)
      build:  docker build ; trivy scan ; push to ECR   (on main)
      deploy: update ECS/K8s to new tag ; smoke test     (on main, approval)

THE PRINCIPLES (the lead view):
    - SHIFT LEFT: catch problems early (lint/test in CI, not in prod).
    - FAST FEEDBACK: keep CI minutes, not hours (parallelize, cache).
    - EVERYTHING AUTOMATED + REPEATABLE: no manual deploy steps to forget.
    - SAME ARTIFACT promoted across environments (build once, deploy many).

INTERVIEW ANSWER:
    "CI runs on every PR — lint, format, type check, the test suite with a
    coverage gate, and for AI changes an eval suite — then builds and scans the
    Docker image. Nothing merges that breaks the build. CD promotes that same
    image through environments: auto-deploy to dev and staging, a manual
    approval gate to prod for most teams. I build the artifact once and promote
    it, so what's tested is exactly what ships. The principles are shift-left to
    catch issues early, fast feedback by keeping CI in minutes with caching and
    parallelism, and full automation so there are no manual steps to forget."
'''


# =================================================================================
# SECTION 6: INFRASTRUCTURE AS CODE (IaC)
# =================================================================================
'''
A lead provisions infra reproducibly, not by clicking in a console.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IaC IS:
    Define infrastructure (servers, DBs, networks, IAM) in CODE/config files,
    version-controlled, applied automatically — instead of manual console
    clicks ("ClickOps").

WHY IT MATTERS:
    - REPRODUCIBLE: spin up an identical environment (dev = staging = prod).
    - VERSIONED: infra changes go through Git + review like code; rollback.
    - AUDITABLE: the code IS the documentation of what exists.
    - NO DRIFT / "snowflake servers": no undocumented manual changes.

TOOLS:
    - TERRAFORM: cloud-agnostic, declarative (HCL), the industry standard.
    - AWS CloudFormation / CDK: AWS-native (CDK = define infra in Python/TS).
    - Pulumi: IaC in real languages (Python, TS).
    - Ansible: configuration management (provisioning + config).

DECLARATIVE (Terraform): declare desired state, tool computes the diff and
    applies it (terraform plan -> apply). Same philosophy as K8s.

INTERVIEW POINT:
    "I provision infrastructure as code — Terraform usually — so environments
    are reproducible, version-controlled, reviewed, and auditable, with no
    snowflake servers or config drift from manual console changes. It's
    declarative: I declare the desired state and the tool computes and applies
    the diff, the same reconcile model as Kubernetes. Infra changes go through
    Git and review exactly like application code."
'''


# =================================================================================
# SECTION 7: GIT & COLLABORATIVE WORKFLOWS (the JD names this)
# =================================================================================
'''
The JD explicitly: "Git... collaborative workflows (branching, pull/merge
requests, and code reviews)." As a LEAD you set the team's Git workflow.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BRANCHING STRATEGIES (know the trade-offs — a lead chooses one):

    TRUNK-BASED DEVELOPMENT (modern default for CD):
        Everyone commits to main (trunk) frequently via SHORT-LIVED feature
        branches (hours/day), merged fast behind feature flags. Enables true CI/CD.
        Pros: minimal merge hell, fast integration. Needs strong tests + flags.

    GITHUB FLOW (simple):
        main is always deployable; branch -> PR -> review -> merge -> deploy.
        Great for web apps / continuous deployment. Simple.

    GIT FLOW (heavier, release-based):
        main + develop + feature/release/hotfix branches. Good for versioned
        releases / multiple supported versions; often overkill for SaaS.

    MY DEFAULT: trunk-based or GitHub Flow with short-lived branches + PRs +
    CI gates + feature flags. Avoid long-lived branches (merge pain, late
    integration).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PR WORKFLOW (the daily collaboration loop):
    1. Branch off main (feature/JIRA-123-add-x).
    2. Commit with clear messages (conventional commits: feat:, fix:, etc.).
    3. Open a PR — small + focused (easy to review). PR template + description.
    4. CI runs (lint/test/build); reviewers review (Phase 8 code review).
    5. Address feedback; squash/rebase to keep history clean.
    6. Merge (squash merge for a clean linear history is common).
    7. Branch protection: require passing CI + approvals before merge to main.

KEY GIT MECHANICS A LEAD KNOWS:
    - merge vs rebase: merge preserves history (merge commits); rebase makes a
      linear history (rewrites commits — don't rebase shared/public branches).
    - Resolving conflicts; cherry-pick (port a fix); revert (undo safely);
      tags for releases; .gitignore hygiene; never commit secrets.
    - PROTECTED main: no direct pushes; PR + CI + review required.

INTERVIEW ANSWER:
    "As a lead I'd run trunk-based development or GitHub Flow — short-lived
    feature branches off main, PRs with CI gates and review, merged quickly,
    with feature flags to decouple deploy from release. I avoid long-lived
    branches because they cause merge pain and late integration. The daily loop
    is: branch, small focused commits with clear messages, a small PR, CI runs
    lint and tests, review, address feedback, then squash-merge for clean
    history — with branch protection requiring passing CI and approvals before
    anything hits main. On mechanics, I use rebase for a linear local history
    but never rebase shared branches, and merge to preserve history; revert to
    undo safely, cherry-pick to port fixes, and never commit secrets."
'''


# =================================================================================
# SECTION 8: MONITORING, LOGGING & OPERATIONS (Day-2)
# =================================================================================
'''
Shipping is day 1; OPERATING is day 2 — where leads earn their keep.
(Overlaps Phase 7 observability; here it's the cloud/ops angle.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE THREE PILLARS OF OBSERVABILITY:
    - METRICS: numeric time-series (latency p50/p95/p99, QPS, error rate, CPU,
      memory, queue depth). Tools: CloudWatch, Prometheus + Grafana.
    - LOGS: structured (JSON) event records with a request/correlation ID so
      you can trace one request across services. Tools: CloudWatch Logs, ELK,
      Loki.
    - TRACES: distributed tracing — follow ONE request across services with
      timing per hop. Tools: OpenTelemetry, X-Ray, Jaeger. (+ LangSmith for LLM).

THE FOUR GOLDEN SIGNALS (what to alert on): LATENCY, TRAFFIC, ERRORS,
    SATURATION (how full the system is).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALERTING + ON-CALL:
    - Alert on SYMPTOMS users feel (high error rate, latency) not just causes;
      avoid alert fatigue (too many noisy alerts -> ignored).
    - SLO / SLA / SLI: SLI = a measured indicator (e.g., % requests < 500ms);
      SLO = the target (99.9%); SLA = the contractual promise. Error budgets
      balance reliability vs feature velocity.
    - RUNBOOKS for common incidents; blameless POST-MORTEMS to learn.

RELIABILITY OPS:
    - Health checks + auto-restart + autoscaling (self-healing).
    - Backups + tested restore + DR plan.
    - Graceful degradation (Phase 7): a partial answer beats an outage.

INTERVIEW ANSWER:
    "Operating is where I focus as a lead. I instrument the three pillars:
    metrics like p95 latency, error rate, and saturation in CloudWatch or
    Prometheus-Grafana; structured logs with a correlation ID to trace a
    request; and distributed tracing with OpenTelemetry plus LangSmith for the
    LLM steps. I alert on the four golden signals — latency, traffic, errors,
    saturation — on user-facing symptoms to avoid alert fatigue, and I define
    SLOs with error budgets to balance reliability against feature velocity.
    Operationally: health checks with auto-restart and autoscaling for
    self-healing, tested backups and a DR plan, runbooks, and blameless
    post-mortems so incidents become improvements."
'''


# =================================================================================
# SECTION 9: COST, SECURITY & SCALING IN THE CLOUD
# =================================================================================
'''
Lead-level cloud concerns: don't blow the budget, don't get breached, scale right.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLOUD COST MANAGEMENT (FinOps):
    - RIGHT-SIZE: don't over-provision; match instance size to actual load.
    - AUTOSCALE: scale to zero / down off-peak (serverless, HPA).
    - SPOT / preemptible instances for fault-tolerant batch (cheap).
    - RESERVED / savings plans for steady baseline load.
    - For AI: the LLM tokens are often the dominant cost (Phase 7) — cache,
      route to smaller models, cap tokens. GPU instances are expensive — use
      managed inference unless volume justifies self-hosting.
    - TAG resources + cost dashboards + budget alerts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLOUD SECURITY (beyond Phase 3/7 app security):
    - IAM least privilege; roles not long-lived keys; rotate credentials.
    - NETWORK: VPC, private subnets for DBs (not public), security groups,
      no public S3 buckets.
    - ENCRYPTION: at rest (KMS) + in transit (TLS).
    - SECRETS: Secrets Manager / Key Vault, never in code/env-in-repo.
    - SCAN: image vulnerability scanning, dependency scanning (SCA).
    - Shared responsibility model (Section 3): misconfig is the top risk.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCALING IN THE CLOUD (ties Phases 1, 4, 7):
    - HORIZONTAL: stateless app + load balancer + autoscaling group / HPA.
    - STATE OUTSIDE the app (DB, cache, object store) so app scales freely.
    - DATA tier: read replicas, caching, partition, shard (Phase 4 order).
    - MULTI-AZ for HA; multi-region for DR / global latency.
    - ASYNC + queues to absorb spikes (smooth bursty load to workers).

INTERVIEW ANSWER:
    "On cost I right-size, autoscale down off-peak, use spot instances for
    fault-tolerant batch and savings plans for steady load, and tag resources
    with budget alerts. For AI the LLM tokens usually dominate cost, so caching
    and model routing matter more than infra. On security: IAM least privilege
    with roles not keys, databases in private subnets, encryption at rest and
    in transit, secrets in a manager, and image and dependency scanning —
    remembering that misconfiguration is the top cloud risk under the shared
    responsibility model. For scaling I keep the app stateless behind a load
    balancer with autoscaling, push state to the DB, cache, and object store,
    scale the data tier with replicas and caching, run multi-AZ for HA, and use
    queues to absorb traffic spikes."
'''


# =================================================================================
# SECTION 10: DEPLOYING LLM/AI WORKLOADS SPECIFICALLY
# =================================================================================
'''
The AI-deployment angle — your differentiator and directly relevant to the role.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MANAGED LLM API vs SELF-HOSTED MODEL (the deployment decision):
    MANAGED (Bedrock / Azure OpenAI / Vertex): no infra, best models, pay per
        token; data leaves to the provider (or stays in your cloud region with
        Bedrock/Azure OpenAI). Fastest path.
    SELF-HOSTED (open model on GPUs): data control, no per-token cost, custom
        models; BUT GPU infra + ops + serving stack (vLLM, TGI) + lower quality
        usually. Justified at high volume or strict data control.

SERVING SELF-HOSTED MODELS (if asked):
    - Inference servers: vLLM, TGI (Text Generation Inference), Triton.
    - GPU autoscaling is hard + costly (slow cold starts, expensive idle).
    - Batching requests to use the GPU efficiently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AI-SPECIFIC DEPLOYMENT CONCERNS:
    - The app tier is I/O-bound (waiting on the LLM) -> async + modest CPU, not
      GPU (unless self-hosting models). Don't over-provision the API tier.
    - LLM latency is high + variable -> generous timeouts, streaming, async jobs.
    - Provider rate limits -> the LLM gateway (Phase 7) + multiple keys/regions.
    - LLMOps: version prompts (Lesson 25), models, and eval sets; canary new
      prompt/model versions; monitor quality (faithfulness) + cost + latency in
      prod; eval gates in CI.
    - Data residency / compliance -> Bedrock or Azure OpenAI in-region, or
      self-host, for regulated data.

MLOps vs LLMOps (a sharp point):
    Classic MLOps = train/version/deploy/monitor YOUR models (data + retraining
    pipelines, drift). LLMOps with API models shifts focus to PROMPTS, RAG,
    evaluation, cost, and guardrails rather than training — you're orchestrating
    a model you don't train.

INTERVIEW ANSWER:
    "For LLM workloads the first decision is managed versus self-hosted. I
    default to managed — Bedrock or Azure OpenAI — for best models and zero GPU
    ops, keeping data in-region for compliance; I self-host an open model on
    GPUs with a server like vLLM only at high volume or strict data-control
    needs, since GPU autoscaling is costly and slow. The API tier itself is
    I/O-bound waiting on the model, so it's async on modest CPU, not GPU — I
    don't over-provision it. I plan for high, variable latency with streaming,
    generous timeouts, and async jobs, and handle rate limits with an LLM
    gateway across multiple keys. Operationally it's LLMOps, not classic MLOps:
    I version prompts, models, and eval sets, canary new versions, and monitor
    faithfulness, cost, and latency in production with eval gates in CI."
'''


# =================================================================================
# SECTION 11: INTERVIEW Q&A (lead-level)
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1. Container vs VM?
A:  "VM virtualizes hardware with a full guest OS — heavy, slow boot. Container
    shares the host kernel, isolates the process — light, seconds to start. Many
    containers per host."

Q2. What does Kubernetes give you?
A:  "Scaling, self-healing, rolling updates, networking. Pods, Deployments
    (desired replicas + rollout), Services (stable LB endpoint), Ingress, HPA
    autoscaling, liveness/readiness probes. Declarative reconcile model."

Q3. Liveness vs readiness probe?
A:  "Liveness — is it alive; fail restarts the pod. Readiness — ready for
    traffic; fail stops routing without restarting. Both back a /health
    endpoint and enable zero-downtime deploys."

Q4. How do you deploy with zero downtime?
A:  "Rolling update with readiness probes, graceful shutdown on SIGTERM to
    finish in-flight requests, and backward-compatible migrations
    (expand-then-contract). Canary or blue-green for risky changes."

Q5. Walk me through your CI/CD.
A:  "CI on every PR: lint, type check, tests with coverage gate, build and scan
    the image. CD promotes the same image — auto to dev/staging, approval gate
    to prod. Build once, promote many."

Q6. What is IaC and why?
A:  "Infrastructure defined in versioned code (Terraform) — reproducible,
    reviewable, auditable, no config drift. Declarative reconcile like K8s."

Q7. Which branching strategy and why?
A:  "Trunk-based or GitHub Flow — short-lived branches, PRs, CI gates, feature
    flags. Avoids long-lived-branch merge pain and enables CI/CD."

Q8. How do you monitor a production service?
A:  "Three pillars — metrics, logs with correlation IDs, distributed traces plus
    LangSmith for LLM steps. Alert on the four golden signals; SLOs with error
    budgets; runbooks and blameless post-mortems."

Q9. Managed LLM API or self-hosted?
A:  "Managed (Bedrock/Azure OpenAI) for best models and zero GPU ops, in-region
    for compliance. Self-host an open model with vLLM only at high volume or
    strict data control — GPU ops are costly."

Q10. How do you control cloud + AI cost?
A:  "Right-size, autoscale down, spot for batch, savings plans for baseline. For
    AI, tokens dominate — cache, route to smaller models, cap tokens. Tag +
    budget alerts."

Q11. How do you secure cloud infra?
A:  "IAM least privilege with roles, private subnets for DBs, encryption at rest
    and in transit, secrets in a manager, image/dependency scanning. Misconfig
    is the top risk under shared responsibility."

Q12. How is LLMOps different from MLOps?
A:  "Classic MLOps centers on training, versioning, and drift of your own
    models. LLMOps with API models shifts to prompts, RAG, evaluation, cost, and
    guardrails — orchestrating a model you don't train."
'''


# =================================================================================
# SECTION 12: GOLDEN LESSONS
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CONTAINERS = PACKAGE ONCE, RUN ANYWHERE. Layer order for caching, slim base,
   non-root, multi-stage. Many containers per host (vs heavy VMs).
2. K8S = SCALE + SELF-HEAL + ROLLING UPDATES. Pod/Deployment/Service/Ingress/HPA
   + liveness/readiness probes. Declarative reconcile. Use ECS/Cloud Run if simpler.
3. KNOW ONE CLOUD DEEPLY (AWS), MAP THE OTHERS. IAM least privilege + shared
   responsibility = the security fundamentals.
4. ZERO-DOWNTIME DEPLOY: rolling + readiness + graceful shutdown + backward-
   compatible migrations. Canary/blue-green for risk.
5. CI/CD: shift left, build once + promote, automate everything, fast feedback.
6. IaC (Terraform): reproducible, versioned, auditable infra — no ClickOps.
7. GIT: trunk-based/GitHub Flow, short branches, PRs + CI gates + protected main,
   feature flags. Never commit secrets; rebase only local.
8. OPERATE (Day-2): metrics+logs+traces, four golden signals, SLOs + error
   budgets, runbooks, blameless post-mortems, self-healing.
9. COST + SECURITY are LEAD concerns: right-size/autoscale/cache; least
   privilege, encryption, secrets manager, scanning.
10. AI DEPLOY: managed LLMs default (in-region for compliance), self-host only
    at scale; API tier is I/O-bound (async, not GPU); LLMOps = prompts/RAG/eval/
    cost/guardrails, version + canary + monitor quality.

ONE-LINE CHEAT SHEET:
    Docker (cache layers, slim, non-root) -> registry -> K8s/ECS (Deployment +
    Service + HPA + probes) behind a load balancer.
    Deploy: rolling + readiness + graceful shutdown + safe migrations; canary for risk.
    CI/CD: lint+type+test+coverage+scan -> build once -> promote dev/stg/prod.
    IaC: Terraform (declarative, versioned). Git: trunk-based + PR + protected main.
    Operate: metrics/logs/traces + golden signals + SLOs + post-mortems.
    Cloud: AWS deep; IAM least-privilege; encrypt; secrets manager; right-size + cache.
    AI: managed LLM (in-region) default; async I/O tier; LLMOps = prompt/RAG/eval/cost.
'''


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 9 — CLOUD & DEVOPS")
    print("=" * 70)
    print()
    print("DEPLOY PATH: Docker image -> registry -> K8s/ECS (Deployment+Service")
    print("  +HPA+probes) behind a load balancer. Rolling update = zero downtime.")
    print()
    print("CONTAINER vs VM: shares host kernel (light, seconds) vs full guest OS.")
    print("K8S: Pod/Deployment/Service/Ingress/HPA + liveness/readiness probes.")
    print()
    print("CI/CD: lint+type+test+coverage+scan -> build ONCE -> promote dev/stg/prod.")
    print("IaC: Terraform (reproducible, versioned). No ClickOps.")
    print("GIT: trunk-based/GitHub Flow + short branches + PR + protected main.")
    print()
    print("OPERATE: metrics+logs+traces, 4 golden signals, SLOs + post-mortems.")
    print("CLOUD: know AWS deep; IAM least-privilege; shared responsibility.")
    print()
    print("AI DEPLOY: managed LLM (in-region) default; API tier is I/O-bound")
    print("  (async, not GPU); LLMOps = prompts/RAG/eval/cost/guardrails.")
    print()
    print("=" * 70)
    print("Remaining (low priority): Phase 10 — Mock Interview & Consolidation.")
    print("=" * 70)
