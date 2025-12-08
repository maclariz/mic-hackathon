# mic-hackathon 2025

## Project Overview

**Author:** Ankit Shrivastava

**Description:** A short description of the research project.

- This project aims to explore and implement various experiments related to **mic-hackathon**.

## Instructions

### Follow the method to push/pull your changes

- **Getting started**

```sh
git pull origin main
```

- **Create your branch for your task (choose a clear name) and follow the workflow locally**

```sh
git checkout -b my-task-branch

git add .
git commit -m "Describe what you did."
```

- **push to remote for the first time**

```sh
git push -u origin my-task-branch
```

- **after that, push usually**

```sh
git push
```
- **If you want to get updates from main into your branch**

```sh
# Update your branch with the newest main changes
git pull origin main --rebase
```

### Directory Structure

```plaintext
resources/
├── old_codes/           # Archived scripts
├── reference_work/      # Similar projects

documents/
├── project_brief.md     # Summary of the project
├── notes/               # Research notes
│   ├── research_rev.md      # Single file containing the literature review
│   ├── bibliography.bib     # Single file for insights

experiment/
├── codes/
├── data/
│   ├── raw/             # Raw input data
│   └── processed/       # Processed data
├── results/
│   └── /
│       └── logs/        # Subexperiment logs

talks/                   # Presentations, abstracts
```
