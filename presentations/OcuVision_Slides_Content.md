# OcuVision Project Presentation — All Slides

Download the PowerPoint: [`presentations/OcuVision_Project_Presentation.pptx`](../presentations/OcuVision_Project_Presentation.pptx)

---

## Slide 1 — Title
**OCUVISION**  
AI-Powered Retinal Disease Screening Platform  
Project Presentation · Backend & Inference Architecture

## Slide 2 — Agenda
1. Problem & clinical need  
2. OcuVision solution overview  
3. System architecture  
4. AI inference engines  
5. Feature extraction pipeline  
6. Supported diagnoses  
7. Backend API & data model  
8. Clinical workflow & demo path  
9. Tech stack & next steps

## Slide 3 — The Problem
Retinal disease is treatable when caught early — but screening capacity is limited.
- **Delayed detection** — DR, glaucoma, AMD progress silently
- **Specialist bottleneck** — uneven ophthalmology access
- **Inconsistent triage** — hard to scale manual fundus review

## Slide 4 — The Solution
Upload fundus → AI diagnosis (severity, confidence, notes) → patient record  
Flow: Capture → Inference (MATLAB / Python) → Clinical output → Record & review  
Design goal: assist triage, not replace specialists. Severities: Healthy · Moderate · Critical

## Slide 5 — Architecture
Client/UI → Node.js Express API → Inference (MATLAB / Python) → Supabase Postgres

## Slide 6 — Multi-Engine AI
- **MATLAB Bagged Trees** — 24 handcrafted features, `.mat` Classification Learner export
- **Python ResNet50 CNN** — end-to-end deep learning, shared JSON schema

## Slide 7 — Feature Pipeline
Image → 512×512 → CLAHE → LBP / GLCM / intensity / vessel metrics → 24 features → predictFcn

## Slide 8 — Diagnoses
Healthy · Diabetic Retinopathy · Glaucoma / Early Glaucoma · AMD · Cataract indicators

## Slide 9 — API
Auth (`/api/login` …) · Analyze & Models (`/api/analyze`, `/api/models`) · Records CRUD

## Slide 10 — Data Model
`users` (doctor/nurse/intern) · `records` (result, severity, fundus_image, features JSONB) · RLS + service role

## Slide 11 — Analyze Flow
POST image → resolve model → python or MATLAB runner → marker JSON → normalize response → optional save record

## Slide 12 — Tech Stack
Node/Express · MATLAB + Python ResNet50 · Supabase · MATLAB Compiler deploy path · shared I/O schema

## Slide 13 — Demo Path
Login → Select model → Upload fundus → Review output → Save OCU-#### record → Health check

## Slide 14 — Closing
OcuVision — faster retinal triage, structured clinical output, multi-engine AI. Thank you.
