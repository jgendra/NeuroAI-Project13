#!/usr/bin/env python3
"""
NeuroAI Project 13 — one-shot GitHub roadmap provisioner
========================================================
Run this ONCE. It creates, in your repo / user account:
  - a GitHub Project (v2)
  - custom fields: Phase, Workflow (Kanban), Priority, Start (date), End (date)
  - labels + milestones (your 3 deadlines)
  - ~34 issues (tasks) with notes, each added to the project with all field values set

It CANNOT create the view tabs (Roadmap / Board / Table) — GitHub's API has no
mutation for views. After running, follow SETUP_INSTRUCTIONS.md (5 min of clicks).
Re-running is safe-ish: it skips labels/milestones/issues/fields that already exist.

USAGE
-----
1. Create a token at https://github.com/settings/tokens
     - "Tokens (classic)" -> Generate new token (classic)
     - Scopes: tick  [x] repo   and   [x] project
2. Run:
    - Paste in terminal: 
    1st: set GITHUB_TOKEN=ghp_49H... (your token)
    Then: python3 setup_roadmap.py
   (or just `python3 setup_roadmap.py` and paste the token when prompted)

Standard library only, no pip install needed.
"""

import os
import sys
import json
import getpass
import urllib.request
import urllib.error

# ----------------------------------------------------------------------------
# CONFIG  — edit these two lines if your repo path is different
# ----------------------------------------------------------------------------
REPO_OWNER = "jgendra"
REPO_NAME = "NeuroAI-Project13"
PROJECT_TITLE = "NeuroAI — Information Decomposition in Task-Trained RNNs"

API = "https://api.github.com"
GQL = "https://api.github.com/graphql"

# ----------------------------------------------------------------------------
# Low-level HTTP helpers (stdlib only)
# ----------------------------------------------------------------------------
def _http(method, url, headers, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt else {})
    except urllib.error.HTTPError as e:
        txt = e.read().decode()
        try:
            parsed = json.loads(txt)
        except Exception:
            parsed = {"raw": txt}
        return e.code, parsed


def rest(method, path, body=None):
    return _http(method, API + path, REST_HEADERS, body)


def gql(query, variables=None):
    status, data = _http("POST", GQL, GQL_HEADERS,
                         {"query": query, "variables": variables or {}})
    if status != 200 or "errors" in data:
        raise RuntimeError("GraphQL error (%s):\n%s"
                           % (status, json.dumps(data.get("errors", data), indent=2)))
    return data["data"]


# ----------------------------------------------------------------------------
# Static data
# ----------------------------------------------------------------------------
LABELS = {
    "stage-0-setup":        ("ededed", "Setup & planning"),
    "stage-1-training":     ("1f6feb", "RNN training"),
    "stage-2-infotheory":   ("2da44e", "Information theory"),
    "stage-3-stats":        ("d4a72c", "Statistics"),
    "stage-4-deliverables": ("8250df", "Deliverables"),
    "priority-high":        ("d73a4a", "Critical path"),
    "priority-medium":      ("fb8500", "Important"),
    "priority-low":         ("c5c5c5", "Stretch / optional"),
    "stretch-goal":         ("e4e669", "Optional / stretch"),
    "milestone":            ("0e8a16", "Key deadline"),
}

MILESTONES = {
    "Final roadmap submission": ("2026-06-08T17:00:00Z", "Submit final roadmap to professors."),
    "Midway presentation":      ("2026-06-17T17:00:00Z", "Mandatory non-graded midway presentation."),
    "Final presentation":       ("2026-07-15T17:00:00Z", "Graded final presentation + individual Q&A."),
}

# single-select fields: name -> [(option, color, description)]
SELECT_FIELDS = {
    "Phase": [
        ("0 \u00b7 Setup & Planning",        "GRAY",   "Planning, setup, roadmap"),
        ("1 \u00b7 RNN Training",            "BLUE",   "CTRNN + training conditions"),
        ("2 \u00b7 Info-Theoretic Analysis", "GREEN",  "PID / \u03a6ID / MI"),
        ("3 \u00b7 Statistical Analysis",    "YELLOW", "Permutation, bootstrap, effects"),
        ("4 \u00b7 Deliverables",            "PURPLE", "Presentations, note, repo"),
    ],
    "Workflow": [
        ("To Do",   "GRAY",   "Not started"),
        ("Doing",   "BLUE",   "In progress"),
        ("Review",  "YELLOW", "Under review"),
        ("Done",    "GREEN",  "Completed"),
    ],
    "Priority": [
        ("High",    "RED",    "Critical path"),
        ("Medium",  "ORANGE", "Important"),
        ("Low",     "GRAY",   "Stretch / optional"),
    ],
}
DATE_FIELDS = ["Start", "End"]

P0 = "0 \u00b7 Setup & Planning"
P1 = "1 \u00b7 RNN Training"
P2 = "2 \u00b7 Info-Theoretic Analysis"
P3 = "3 \u00b7 Statistical Analysis"
P4 = "4 \u00b7 Deliverables"


def task(title, phase, workflow, start, end, priority, labels, body,
         milestone=None):
    return dict(title=title, phase=phase, workflow=workflow, start=start,
                end=end, priority=priority, labels=labels, body=body,
                milestone=milestone)


TASKS = [
    # ---------------- Stage 0 : Setup & Planning ----------------
    task("Draft roadmap, goals & research questions", P0, "Done",
         "2026-05-27", "2026-05-29", "High",
         ["stage-0-setup", "priority-high"],
         "DONE. First draft (the proposal): motivation, Q1-Q4, experiment design "
         "(2 tasks x 3 conditions x 5 seeds = 30 RNNs), metrics and references. "
         "This is the basis for the GitHub roadmap.",
         milestone="Final roadmap submission"),

    task("Write & send project proposal to professors", P0, "Done",
         "2026-05-27", "2026-05-29", "High",
         ["stage-0-setup", "priority-high"],
         "DONE. Proposal emailed to the NeuroAI team. Confirms scope: a controlled "
         "methods-and-mechanisms study in artificial RNNs (no claim about biological data)."),

    task("Part 0 - Agree work distribution (3 members)", P0, "Doing",
         "2026-05-29", "2026-06-02", "High",
         ["stage-0-setup", "priority-high"],
         "Assign owners per stage. Suggested split:\n"
         "- A: RNN training & infrastructure (Stage 1)\n"
         "- B: information-theory pipeline (Stage 2)\n"
         "- C: statistics + writing/presentations (Stages 3-4)\n"
         "Everyone reviews each other's parts. Set the 'Assignees' on each issue accordingly."),

    task("Set up repo structure + Python environment", P0, "To Do",
         "2026-05-30", "2026-06-04", "High",
         ["stage-0-setup", "priority-high"],
         "Repo layout (src/, notebooks/, results/, figures/). Environment: PyTorch, "
         "NeuroGym, phyid, JIDT (needs Java + JPype), numpy/scipy. Pin versions in "
         "environment.yml — the course explicitly requires a reproducible GitHub repo."),

    task("Consolidate key literature (1 note per paper)", P0, "To Do",
         "2026-05-30", "2026-06-07", "Medium",
         ["stage-0-setup", "priority-medium"],
         "PID (Williams & Beer 2010), \u03a6ID (Mediano/Rosas 2021), Luppi 2022 "
         "(sensory=redundancy, association=synergy), Mante 2013, Yang 2019, "
         "Sussillo & Barak 2013. One short note per paper in the repo wiki."),

    task("Finalize & submit roadmap to professors (M1)", P0, "To Do",
         "2026-06-06", "2026-06-08", "High",
         ["stage-0-setup", "priority-high", "milestone"],
         "DEADLINE 8 June. Export this board (screenshot of the Roadmap view + a short "
         "written plan) and submit. Fold in any early feedback first.",
         milestone="Final roadmap submission"),

    # ---------------- Stage 1 : RNN Training ----------------
    task("Finalize CTRNN architecture", P1, "To Do",
         "2026-06-01", "2026-06-03", "High",
         ["stage-1-training", "priority-high"],
         "CTRNN, 1 recurrent layer, 80 units; tanh recurrent, linear in/out. "
         "tau=100ms, dt=20ms (dt/tau=0.2). Euler: "
         "h(t+dt)=h(t)+(dt/tau)[-h + W_rec.tanh(h) + W_in.u + b]. "
         "Init: W_rec orthogonal gain 1.0; W_in/W_out/W_pred Xavier; biases 0. "
         "Fully connected, no Dale's law."),

    task("Get NeuroGym tasks running", P1, "To Do",
         "2026-06-02", "2026-06-04", "High",
         ["stage-1-training", "priority-high"],
         "PerceptualDecisionMaking-v0 (control, low integration) and "
         "ContextDecisionMaking-v0 (high integration). Override timing with "
         "env_kwargs={'dt':20}. Inputs: 3 ch (PerceptualDM) / 7 ch (ContextDM); 3 actions out."),

    task("Implement training loop (BPTT)", P1, "To Do",
         "2026-06-03", "2026-06-07", "High",
         ["stage-1-training", "priority-high"],
         "Euler integration + BPTT (O(N.T) compute, fine at this scale). Adam lr=1e-3, "
         "batch 16, grad-norm clip 1.0, <=20k trials, early stopping on resampled "
         "validation. Cross-entropy masked to the decision period."),

    task("Train vanilla supervised baseline (both tasks)", P1, "To Do",
         "2026-06-06", "2026-06-09", "High",
         ["stage-1-training", "priority-high"],
         "Condition 1: L = CrossEntropy. Confirm convergence on both tasks; this is the "
         "baseline. Compute note: an 80-unit CTRNN trains in minutes on a laptop CPU."),

    task("Implement activity-regularized condition", P1, "To Do",
         "2026-06-08", "2026-06-10", "High",
         ["stage-1-training", "priority-high"],
         "Condition 2: L = CE + lambda.mean(h^2), lambda~1e-4 (efficient-coding / "
         "metabolic-cost analog). Tune lambda."),

    task("Implement predictive-loss condition", P1, "To Do",
         "2026-06-09", "2026-06-11", "High",
         ["stage-1-training", "priority-high"],
         "Condition 3: L = CE + mu.MSE(W_pred.h(t), u(t+1)), mu~0.1 (predictive-coding "
         "analog). W_pred is a training-only linear head, dropped at evaluation."),

    task("Match accuracy across conditions", P1, "To Do",
         "2026-06-10", "2026-06-12", "High",
         ["stage-1-training", "priority-high"],
         "Tune lambda and mu so task accuracy matches the baseline BEFORE comparing "
         "information structure. Otherwise any PID difference could just reflect a "
         "performance gap, not a change in how info is organized."),

    task("Run full training sweep (30 RNNs)", P1, "To Do",
         "2026-06-11", "2026-06-14", "High",
         ["stage-1-training", "priority-high"],
         "2 tasks x 3 conditions x 5 seeds = 30 networks. Save weights + hidden-state "
         "activations h(t) per trial. Laptop-feasible; batch overnight (~a few hours total)."),

    # ---------------- Stage 2 : Information-Theoretic Analysis ----------------
    task("PID vs \u03a6ID deep-dive + estimator choice", P2, "To Do",
         "2026-06-08", "2026-06-12", "High",
         ["stage-2-infotheory", "priority-high"],
         "Pin down what each measures (PID: redundant/unique/synergistic decomposition of "
         "sources about a target; \u03a6ID: dynamical/temporal extension). Choose redundancy "
         "lattice + estimator. Likely DROP linear Fisher info (redundant) and treat \u03a6ID "
         "as a stretch goal (computationally expensive)."),

    task("Set up & validate phyid + JIDT", P2, "To Do",
         "2026-06-12", "2026-06-16", "High",
         ["stage-2-infotheory", "priority-high"],
         "Install phyid (Imperial-MIND-lab) + JIDT (Java/JPype). Validate on a toy system "
         "with KNOWN synergy/redundancy before trusting it on RNN data."),

    task("Define coarse-graining (80 -> 2 subpopulations)", P2, "To Do",
         "2026-06-14", "2026-06-16", "Medium",
         ["stage-2-infotheory", "priority-medium"],
         "PID over all 80 units is intractable. Coarse-grain into 2 (max 2-4) functional "
         "subpopulations to keep PID/\u03a6ID tractable. Document the grouping rule."),

    task("Compute PID across 30 networks", P2, "To Do",
         "2026-06-16", "2026-06-22", "High",
         ["stage-2-infotheory", "priority-high"],
         "PID with stimulus/decision as target on h(t) at a fixed timestep, per network. "
         "Gives redundancy/unique/synergy per condition x task -> the core of Q1-Q4."),

    task("Pairwise MI between units (KSG) sanity check", P2, "To Do",
         "2026-06-18", "2026-06-22", "Medium",
         ["stage-2-infotheory", "priority-medium"],
         "KSG estimator for pairwise MI between units as a correlation-structure sanity "
         "check. (MINE optional - heavier, needs training a net per estimate.)"),

    task("[STRETCH] \u03a6ID dynamical analysis", P2, "To Do",
         "2026-06-23", "2026-06-29", "Low",
         ["stage-2-infotheory", "priority-low", "stretch-goal"],
         "\u03a6ID with h(t+tau) as target -> does the network integrate information over "
         "time synergistically? Compute-heavy; only if time + laptop budget allow."),

    task("[OPTIONAL] Linear Fisher information", P2, "To Do",
         "2026-06-23", "2026-06-27", "Low",
         ["stage-2-infotheory", "priority-low", "stretch-goal"],
         "Kanitscheider 2015 bias-corrected linear Fisher info. Likely redundant with "
         "MI/PID - include only if it adds a distinct angle."),

    # ---------------- Stage 3 : Statistical Analysis ----------------
    task("Permutation tests (Q1-Q3)", P3, "To Do",
         "2026-06-23", "2026-06-27", "High",
         ["stage-3-stats", "priority-high"],
         "Shuffle condition labels to build an empirical null. Needed because PID/\u03a6ID "
         "estimates are bounded, skewed and non-Gaussian, so parametric tests are invalid "
         "(especially with only 5 seeds per condition)."),

    task("Bootstrap CIs + effect sizes", P3, "To Do",
         "2026-06-25", "2026-06-29", "High",
         ["stage-3-stats", "priority-high"],
         "Bootstrap resampling for confidence intervals; Cohen's d for effect sizes. "
         "Report alongside p-values."),

    task("Multiple-comparison correction", P3, "To Do",
         "2026-06-27", "2026-06-30", "Medium",
         ["stage-3-stats", "priority-medium"],
         "Bonferroni (or Holm) across planned comparisons "
         "(e.g. condition 1 vs 2 on ContextDM)."),

    task("Q4 interaction analysis (loss x task)", P3, "To Do",
         "2026-06-29", "2026-07-03", "High",
         ["stage-3-stats", "priority-high"],
         "Test whether loss effects are larger on high-integration ContextDM than on "
         "PerceptualDM. A significant interaction = synergy emerges when the task demands "
         "integration (the deepest result the project could produce)."),

    task("Synthesize results + build final figures", P3, "To Do",
         "2026-07-01", "2026-07-06", "High",
         ["stage-3-stats", "priority-high"],
         "Final figures: redundancy-synergy by condition x task, the interaction plot, MI "
         "sanity. Map each back to Q1-Q4 with expected-vs-observed."),

    # ---------------- Stage 4 : Deliverables ----------------
    task("Prepare midway presentation slides", P4, "To Do",
         "2026-06-13", "2026-06-16", "High",
         ["stage-4-deliverables", "priority-high", "milestone"],
         "Short deck: motivation, Q1-Q4, methods, progress (training + analysis pipeline), "
         "planned analyses. Non-graded but mandatory.",
         milestone="Midway presentation"),

    task("Midway presentation (M2)", P4, "To Do",
         "2026-06-17", "2026-06-17", "High",
         ["stage-4-deliverables", "priority-high", "milestone"],
         "DEADLINE 17 June. Present progress and collect professor feedback.",
         milestone="Midway presentation"),

    task("Incorporate midway feedback", P4, "To Do",
         "2026-06-18", "2026-06-22", "Medium",
         ["stage-4-deliverables", "priority-medium"],
         "Adjust scope / analyses based on professor feedback from the midway."),

    task("Write technical note", P4, "To Do",
         "2026-07-01", "2026-07-10", "High",
         ["stage-4-deliverables", "priority-high"],
         "Methods + results technical note (course deliverable alongside the repo). "
         "Clear, reproducible, honest about null/negative results."),

    task("Clean & document GitHub repo", P4, "To Do",
         "2026-07-06", "2026-07-12", "High",
         ["stage-4-deliverables", "priority-high"],
         "README, reproducible scripts, fixed seeds, environment file, instructions to "
         "regenerate every figure. Reproducibility is an explicit course goal."),

    task("Prepare final presentation slides", P4, "To Do",
         "2026-07-08", "2026-07-13", "High",
         ["stage-4-deliverables", "priority-high", "milestone"],
         "Graded deck: emphasize HYPOTHESIS & STUDY DESIGN (the course's grading focus), "
         "results for Q1-Q4, limitations, and prep for individual Q&A.",
         milestone="Final presentation"),

    task("Final presentation (M3)", P4, "To Do",
         "2026-07-15", "2026-07-15", "High",
         ["stage-4-deliverables", "priority-high", "milestone"],
         "DEADLINE 15 July. Graded group presentation + individual questions.",
         milestone="Final presentation"),

    task("Buffer / contingency week", P4, "To Do",
         "2026-07-07", "2026-07-14", "Medium",
         ["stage-4-deliverables", "priority-medium"],
         "Slack for debugging, re-runs or scope cuts. PROTECT this time - info-theory "
         "estimators and accuracy-matching almost always take longer than planned."),
]


# ----------------------------------------------------------------------------
# Provisioning steps
# ----------------------------------------------------------------------------
def ensure_labels():
    print("\n== Labels ==")
    for name, (color, desc) in LABELS.items():
        status, _ = rest("POST", "/repos/%s/%s/labels" % (REPO_OWNER, REPO_NAME),
                          {"name": name, "color": color, "description": desc})
        if status == 201:
            print("  + created label '%s'" % name)
        elif status == 422:
            print("  = label '%s' already exists" % name)
        else:
            print("  ! label '%s' -> HTTP %s" % (name, status))


def ensure_milestones():
    print("\n== Milestones ==")
    status, existing = rest("GET",
                            "/repos/%s/%s/milestones?state=all&per_page=100"
                            % (REPO_OWNER, REPO_NAME))
    by_title = {m["title"]: m["number"] for m in existing} if isinstance(existing, list) else {}
    for title, (due, desc) in MILESTONES.items():
        if title in by_title:
            print("  = milestone '%s' already exists" % title)
            continue
        status, m = rest("POST", "/repos/%s/%s/milestones" % (REPO_OWNER, REPO_NAME),
                         {"title": title, "due_on": due, "description": desc})
        if status == 201:
            by_title[title] = m["number"]
            print("  + created milestone '%s'" % title)
        else:
            print("  ! milestone '%s' -> HTTP %s: %s" % (title, status, m))
    return by_title


def list_existing_issue_titles():
    titles = {}
    page = 1
    while True:
        status, items = rest("GET",
                             "/repos/%s/%s/issues?state=all&per_page=100&page=%d"
                             % (REPO_OWNER, REPO_NAME, page))
        if status != 200 or not isinstance(items, list) or not items:
            break
        for it in items:
            if "pull_request" in it:
                continue
            titles[it["title"]] = it["node_id"]
        if len(items) < 100:
            break
        page += 1
    return titles


def ensure_issues(milestone_numbers):
    print("\n== Issues ==")
    existing = list_existing_issue_titles()
    title_to_node = {}
    for t in TASKS:
        if t["title"] in existing:
            print("  = issue exists: %s" % t["title"])
            title_to_node[t["title"]] = existing[t["title"]]
            continue
        payload = {"title": t["title"], "body": t["body"], "labels": t["labels"]}
        if t["milestone"]:
            num = milestone_numbers.get(t["milestone"])
            if num is not None:
                payload["milestone"] = num
        status, issue = rest("POST", "/repos/%s/%s/issues" % (REPO_OWNER, REPO_NAME), payload)
        if status == 201:
            title_to_node[t["title"]] = issue["node_id"]
            print("  + created issue: %s" % t["title"])
        else:
            print("  ! issue failed: %s -> HTTP %s: %s" % (t["title"], status, issue))
    return title_to_node


def get_ids():
    q = """
    query($owner:String!, $name:String!){
      repositoryOwner(login:$owner){
        id
        ... on ProjectV2Owner { projectsV2(first:100){ nodes{ id title number } } }
      }
      repository(owner:$owner, name:$name){ id }
    }"""
    d = gql(q, {"owner": REPO_OWNER, "name": REPO_NAME})
    owner = d["repositoryOwner"]
    repo = d["repository"]
    if owner is None or repo is None:
        raise RuntimeError("Could not resolve owner/repo. Check REPO_OWNER / REPO_NAME and token scopes.")
    projects = owner.get("projectsV2", {}).get("nodes", []) or []
    return owner["id"], repo["id"], projects


def ensure_project(owner_id, repo_id, existing_projects):
    print("\n== Project ==")
    for p in existing_projects:
        if p["title"] == PROJECT_TITLE:
            print("  = project already exists (#%s)" % p["number"])
            link_project(p["id"], repo_id)
            return p["id"], p["number"]
    q = """
    mutation($ownerId:ID!, $title:String!){
      createProjectV2(input:{ownerId:$ownerId, title:$title}){ projectV2{ id number } }
    }"""
    d = gql(q, {"ownerId": owner_id, "title": PROJECT_TITLE})
    pv = d["createProjectV2"]["projectV2"]
    print("  + created project #%s" % pv["number"])
    link_project(pv["id"], repo_id)
    return pv["id"], pv["number"]


def link_project(project_id, repo_id):
    q = """
    mutation($projectId:ID!, $repoId:ID!){
      linkProjectV2ToRepository(input:{projectId:$projectId, repositoryId:$repoId}){ repository{ id } }
    }"""
    try:
        gql(q, {"projectId": project_id, "repoId": repo_id})
        print("  + linked project to repo")
    except RuntimeError as e:
        # already linked is fine
        print("  = project link note: %s" % str(e).splitlines()[0])


def list_fields(project_id):
    q = """
    query($id:ID!){
      node(id:$id){ ... on ProjectV2 { fields(first:50){ nodes{
        ... on ProjectV2FieldCommon { id name dataType }
        ... on ProjectV2SingleSelectField { id name options{ id name } }
      }}}}
    }"""
    d = gql(q, {"id": project_id})
    return d["node"]["fields"]["nodes"]


def ensure_fields(project_id):
    print("\n== Fields ==")
    nodes = list_fields(project_id)
    by_name = {n["name"]: n for n in nodes if n}
    field_ids = {}                 # field name -> field id
    option_ids = {}                # field name -> {option name -> id}

    # single-select fields
    for fname, opts in SELECT_FIELDS.items():
        if fname in by_name:
            print("  = field '%s' exists" % fname)
            field_ids[fname] = by_name[fname]["id"]
            option_ids[fname] = {o["name"]: o["id"] for o in by_name[fname].get("options", [])}
            continue
        q = """
        mutation($projectId:ID!, $name:String!, $opts:[ProjectV2SingleSelectFieldOptionInput!]){
          createProjectV2Field(input:{projectId:$projectId, dataType:SINGLE_SELECT, name:$name, singleSelectOptions:$opts}){
            projectV2Field{ ... on ProjectV2SingleSelectField { id name options{ id name } } }
          }
        }"""
        opt_input = [{"name": n, "color": c, "description": dsc} for (n, c, dsc) in opts]
        d = gql(q, {"projectId": project_id, "name": fname, "opts": opt_input})
        f = d["createProjectV2Field"]["projectV2Field"]
        field_ids[fname] = f["id"]
        option_ids[fname] = {o["name"]: o["id"] for o in f["options"]}
        print("  + created field '%s'" % fname)

    # date fields
    for fname in DATE_FIELDS:
        if fname in by_name:
            print("  = field '%s' exists" % fname)
            field_ids[fname] = by_name[fname]["id"]
            continue
        q = """
        mutation($projectId:ID!, $name:String!){
          createProjectV2Field(input:{projectId:$projectId, dataType:DATE, name:$name}){
            projectV2Field{ ... on ProjectV2FieldCommon { id name } }
          }
        }"""
        d = gql(q, {"projectId": project_id, "name": fname})
        field_ids[fname] = d["createProjectV2Field"]["projectV2Field"]["id"]
        print("  + created field '%s'" % fname)

    return field_ids, option_ids


def add_item(project_id, content_node_id):
    q = """
    mutation($projectId:ID!, $contentId:ID!){
      addProjectV2ItemById(input:{projectId:$projectId, contentId:$contentId}){ item{ id } }
    }"""
    return gql(q, {"projectId": project_id, "contentId": content_node_id})["addProjectV2ItemById"]["item"]["id"]


def set_value(project_id, item_id, field_id, value):
    q = """
    mutation($projectId:ID!, $itemId:ID!, $fieldId:ID!, $value:ProjectV2FieldValue!){
      updateProjectV2ItemFieldValue(input:{projectId:$projectId, itemId:$itemId, fieldId:$fieldId, value:$value}){
        projectV2Item{ id }
      }
    }"""
    gql(q, {"projectId": project_id, "itemId": item_id, "fieldId": field_id, "value": value})


def populate_items(project_id, field_ids, option_ids, title_to_node):
    print("\n== Adding items to project + setting fields ==")
    for t in TASKS:
        node = title_to_node.get(t["title"])
        if not node:
            print("  ! skipping (no issue node): %s" % t["title"])
            continue
        item_id = add_item(project_id, node)
        set_value(project_id, item_id, field_ids["Phase"],
                  {"singleSelectOptionId": option_ids["Phase"][t["phase"]]})
        set_value(project_id, item_id, field_ids["Workflow"],
                  {"singleSelectOptionId": option_ids["Workflow"][t["workflow"]]})
        set_value(project_id, item_id, field_ids["Priority"],
                  {"singleSelectOptionId": option_ids["Priority"][t["priority"]]})
        set_value(project_id, item_id, field_ids["Start"], {"date": t["start"]})
        set_value(project_id, item_id, field_ids["End"], {"date": t["end"]})
        print("  + %s" % t["title"])


# ----------------------------------------------------------------------------
def main():
    token = os.environ.get("GITHUB_TOKEN", "").strip().strip('"').strip("'")
    if not token:
        # visible input() — paste works reliably in VS Code, unlike getpass
        token = input("Paste your GitHub token (classic, scopes repo+project): ").strip().strip('"').strip("'")
    if not token:
        print("No token provided. Aborting.")
        sys.exit(1)

    global REST_HEADERS, GQL_HEADERS
    REST_HEADERS = {
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "neuroai-roadmap-setup",
    }
    GQL_HEADERS = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "User-Agent": "neuroai-roadmap-setup",
    }

    st, who = rest("GET", "/user")
    if st != 200:
        print("Token rejected (HTTP %s): %s" % (st, who.get("message")))
        print("Regenerate a classic token with 'repo' + 'project' scopes and copy the full ghp_ value.")
        sys.exit(1)
    print("Authenticated as: %s" % who.get("login"))

    print("Target: %s/%s" % (REPO_OWNER, REPO_NAME))
    print("Project title: %s" % PROJECT_TITLE)

    ensure_labels()
    ms = ensure_milestones()
    title_to_node = ensure_issues(ms)
    owner_id, repo_id, existing_projects = get_ids()
    project_id, project_number = ensure_project(owner_id, repo_id, existing_projects)
    field_ids, option_ids = ensure_fields(project_id)
    populate_items(project_id, field_ids, option_ids, title_to_node)

    url = "https://github.com/users/%s/projects/%d" % (REPO_OWNER, project_number)
    print("\n" + "=" * 64)
    print("DONE. Your project: %s" % url)
    print("=" * 64)
    print("""
NEXT (manual, ~5 min) — create the 3 view tabs. The API can't do this.

Open the project, then for each tab use the '+' next to the view tabs:

1) ROADMAP tab  (your main Gantt view)
   - New view -> Layout: Roadmap
   - Top-right of the view -> the date markers/zoom control:
       set "Start date field" = Start  and  "Target date field" = End
   - (optional) Group by: Phase   |   Markers: show Milestones (your 3 deadlines)
   - Zoom: Month
   - Rename the tab to "Roadmap"

2) BOARD tab  (Kanban: To Do / Doing / Review / Done)
   - New view -> Layout: Board
   - Group by: Workflow
   - Rename the tab to "Board"

3) TABLE tab  (by stage, dates vertical — like github/roadmap)
   - New view -> Layout: Table
   - Group by: Phase
   - Sort by: Start (ascending)
   - Show the Start, End, Priority, Workflow columns
   - Rename the tab to "Table"

Tip for the 8 June submission: screenshot the Roadmap tab + export/print.
""")


if __name__ == "__main__":
    main()
