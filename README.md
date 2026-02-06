# Content Style Board

Content Style Board is a local web application that automatically finds high-performing social media posts and displays them as visual, draggable thumbnails on an infinite whiteboard. Posts are grouped by content style (e.g. carousels, reels, threads) to make researching and remixing content formats fast and visual.

This project was built as a hands-on way to practice designing, building, and iterating on a real product from scratch.

---

## Why This Exists

Finding inspiration for social content is slow and manual. Most creators scroll endlessly through profiles, save screenshots, or copy links into notes with no real organization.

Content Style Board removes that friction by:
- Automatically discovering relevant posts
- Generating visual previews (thumbnails, not favicons)
- Letting you organize ideas spatially on a whiteboard

The goal is to make content research feel closer to a visual thinking tool than a list of bookmarks.

---

## Current Features

- Style-based search (e.g. carousels, single-clip reels, multi-clip reels, threads)
- Automatic discovery using web search
- Visual link previews using Open Graph metadata
- Infinite whiteboard canvas with pan and zoom
- Draggable and resizable group boxes
- Ability to delete individual items or entire groups
- Local-first persistence (no database)
- AI-assisted content style tagging

---

## Tech Stack

- Python
- FastAPI
- Brave Search API
- Google Gemini API
- Vanilla HTML, CSS, and JavaScript

---

## Running Locally

1. Clone the repository:
   ```
   git clone https://github.com/JerryG0311/content-style-board.git
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```
   uvicorn src.app:app --reload
   ```

5. Open the app in your browser:
   ```
   http://127.0.0.1:8000
   ```

---

## Notes

- This is a local-first project intended for learning and experimentation.
- Runtime data (board state, unfurl cache) is intentionally excluded from version control.
- The UI and feature set are intentionally kept framework-free.

---

## Project Goals

The goal of this project is not to build a production SaaS or monetize the app.

It exists to practice:
- Breaking a product into smaller problems
- Building backend and frontend pieces together
- Iterating on UI/UX through real usage
- Using Git consistently throughout development