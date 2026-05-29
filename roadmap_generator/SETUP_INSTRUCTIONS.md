# NeuroAI Project 13 — GitHub Roadmap Setup

This sets up a full GitHub Project (v2) roadmap for the project: a **Roadmap (Gantt)**
tab, a **Kanban Board** tab, and a **Table** tab — all populated with 34 tasks, notes,
your 3 deadlines, labels and priorities.

The `setup_roadmap.py` script does ~95% automatically. The only manual part is
**creating the 3 view tabs** (~5 minutes of clicks) — GitHub's API genuinely has no way
to create views, so *no* tool or account connection could automate that step.

---

## 1. Get a token (2 min)

1. Go to **https://github.com/settings/tokens**
2. **Generate new token → Tokens (classic)**
3. Give it a name, set an expiry (30 days is fine)
4. Tick these scopes: **`repo`** and **`project`**
5. Generate, then **copy** the token (starts with `ghp_…`)

## 2. Run the script (1 min)

```bash
# from the folder containing setup_roadmap.py
GITHUB_TOKEN=ghp_yourtokenhere python3 setup_roadmap.py
```

…or just `python3 setup_roadmap.py` and paste the token when prompted.

Pure standard library — no `pip install` needed. Re-running is safe: it skips labels,
milestones, issues and fields that already exist.

> If your repo path differs, edit `REPO_OWNER` / `REPO_NAME` at the top of the script.

When it finishes it prints a link to your new project.

## 3. Create the 3 view tabs (~5 min)

Open the project. Next to the view tabs there's a **`+`**. Add each view:

### Tab 1 — Roadmap (your main Gantt view)
- `+` → **New view** → Layout: **Roadmap**
- In the view’s date control (top-right): **Start date field = `Start`**, **Target date field = `End`**
- *(optional)* Group by **`Phase`**; turn on **Markers → Milestones** to show your 3 deadlines as vertical lines
- Zoom: **Month**
- Double-click the tab name → rename to **Roadmap**

### Tab 2 — Board (Kanban: To Do / Doing / Review / Done)
- `+` → **New view** → Layout: **Board**
- **Group by → `Workflow`**
- Rename the tab to **Board**

### Tab 3 — Table (by stage, like github/roadmap)
- `+` → **New view** → Layout: **Table**
- **Group by → `Phase`**
- **Sort by → `Start`** (ascending)
- Make sure the `Start`, `End`, `Priority`, `Workflow` columns are visible
- Rename the tab to **Table**

---

## What you get

**Custom fields:** `Phase` (5 stages), `Workflow` (Kanban status), `Priority`, `Start`, `End`.

**Milestones (your deadlines):**
- Final roadmap submission — **8 June 2026**
- Midway presentation — **17 June 2026**
- Final presentation — **15 July 2026**

**Stages:**
- 0 · Setup & Planning (6 tasks)
- 1 · RNN Training (8)
- 2 · Info-Theoretic Analysis (7)
- 3 · Statistical Analysis (5)
- 4 · Deliverables (8)

The two completed tasks (proposal + first-draft roadmap) are pre-marked **Done**.

## Notes built into the schedule
- **Compute:** training 30 small 80-unit CTRNNs is laptop-friendly (minutes each, batch
  overnight). The real cost is PID/ΦID — so PID is kept tractable via 2 subpopulations,
  and **ΦID + Fisher info are flagged as stretch/optional**, matching the proposal.
- A **buffer week** before the final presentation is intentionally protected.
- For the **8 June** submission: screenshot the Roadmap tab and export/print it.

## Tweaking later
- Change a date → edit the `Start`/`End` columns in the Table view; the Roadmap updates automatically.
- Assign owners → set **Assignees** on each issue once you’ve agreed the Part 0 split.
- Add/remove a task → either do it in the UI, or edit the `TASKS` list in the script and re-run.
