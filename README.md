---
title: TrapSentinel AI
emoji: 🪲
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# TrapSentinel AI

Smart pest detection and identification system using FastAPI, YOLO, Firebase Realtime Database, Cloudinary, and a static web dashboard.

## Web App

- `/ui` - Dashboard
- `/ui/logs` - Detection logs
- `/ui/access` - Access management
- `/api/status` - API status

## API

- `POST /api/analyze`
- `GET /api/logs`
- `GET /api/logs/{id}`
- `GET /api/dashboard`
- `GET /api/live/latest`