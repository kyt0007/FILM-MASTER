# 🎬 pAssistant — AI Film Generator

An autonomous multi-agent system that turns a one-line idea into a fully assembled short film. Built on Google's **Agent Development Kit (ADK)**, it orchestrates Gemini, Imagen 4, and Veo 3.0 across a conversational 7-step pipeline — pausing only twice for human approval.

---

## Features

- **Conversational director interface** — describe your idea; the agent handles everything else
- **Structured pre-production** — generates film concept → script → detailed shot list automatically
- **Parallel media generation** — character portraits and location reference images rendered simultaneously
- **Image-conditioned video** — each scene image is fed directly into Veo 3.0 for visual consistency
- **Embedded audio** — dialogue and ambient sound generated natively by Veo 3.0 (no post-processing)
- **Automatic assembly** — all clips downloaded from GCS, concatenated, and re-uploaded as a final MP4
- **Surgical regeneration** — fix any individual portrait, scene image, or video without redoing the whole pipeline
- **Quality gate** — every generated image is evaluated by Gemini on 6 dimensions before being accepted

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User / ADK Runner                            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ natural-language messages
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│               Root Agent  (gemini-2.5-flash)                        │
│                                                                     │
│  PRE-PRODUCTION                                                     │
│  ① brainstorm_film_concept ──► FilmConcept (title/genre/style)      │
│  ② write_script            ──► Scene-by-scene script + dialogue     │
│  ③ design_scenes           ──► Shot list (camera/framing/locations) │
│                                                  ⏸ User approves    │
│                 ┌────────────────────────────────────────────┐      │
│  ④ media_gen_agent         ParallelAgent                     │      │
│  │  ├─ character_portrait_agent ──► Imagen 4 Ultra portraits  │      │
│  │  └─ location_image_agent     ──► Gemini location images    │      │
│                 └────────────────────────────────────────────┘      │
│                                                  ⏸ User approves    │
│                                                                     │
│  PRODUCTION                                                         │
│                 ┌────────────────────────────────────────────┐      │
│  ⑤-⑦ production_pipeline_agent   SequentialAgent             │      │
│  │  ├─ scene_images_agent ──► Gemini scene images (per shot)  │      │
│  │  ├─ video_agent        ──► Veo 3.0 clips + embedded audio  │      │
│  │  └─ assemble_agent     ──► Final MP4 (moviepy concat)      │      │
│                 └────────────────────────────────────────────┘      │
│                                                                     │
│  REGENERATION TOOLS (on-demand)                                     │
│  • regenerate_character_portrait(name, new_description)             │
│  • regenerate_scene_image(shot_id, new_prompt)                      │
│  • regenerate_scene_video(shot_id)                                  │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          Google Cloud Storage          ADK Artifacts
          (scene images, videos,        (portraits, location refs,
           final MP4)                    scene images)
```

### State Flow Between Pipeline Steps

```
brainstorm_film_concept
        │ writes: film_concept, title
        ▼
write_script
        │ writes: script, characters
        ▼
design_scenes
        │ writes: shot_list
        ▼
media_gen_agent ──────────────────────────────────┐
        │ writes: character_portraits (artifacts)  │
        │         location_images (artifacts)      │ parallel
        ▼                                          ┘
scene_images_agent
        │ reads:  shot_list, character_portraits, location_images
        │ writes: scene_image_uris (GCS)
        ▼
video_agent
        │ reads:  shot_list, scene_image_uris
        │ writes: video_uri_map, video_uris (GCS)
        ▼
assemble_agent
        │ reads:  video_uri_map
        │ writes: final_film_uri (GCS), final_film_artifact
        ▼
      Final MP4
```

---

## Technologies Used

| Layer | Technology | Role |
|---|---|---|
| Agent Framework | [Google ADK](https://google.github.io/adk-docs/) | LlmAgent, ParallelAgent, SequentialAgent, ToolContext, session state |
| LLM Orchestration | Gemini 2.5 Flash | Root agent reasoning, script/concept/shot design, image evaluation |
| Image Generation | Gemini 2.0 Flash Preview (`gemini-2.0-flash-preview-image-generation`) | Scene images and location references |
| Portrait Generation | Imagen 4 Ultra (`imagen-4.0-ultra`) | Character portraits (falls back to Gemini on failure) |
| Video Generation | Veo 3.0 (`veo-3.0-generate-preview`) | 8-second scene clips with embedded dialogue audio |
| Video Assembly | [moviepy](https://zulko.github.io/moviepy/) ≥ 2.0 | Re-encode clips, concatenate into final film |
| Storage | Google Cloud Storage | Scene images, Veo output videos, final MP4 |
| Artifact Store | ADK Artifact Service | Portrait and scene images passed between pipeline steps |
| Data Validation | Pydantic | FilmConcept, Script, ShotList schemas |
| Cloud Platform | Google Cloud / Vertex AI | All API calls route through Vertex AI |

---

## Pipeline Walkthrough

| Step | Name | What Happens |
|---|---|---|
| ① | Film Concept | Agent expands your idea into title, genre, visual style, mood, 3-act structure, and sound direction. **Pauses for your approval.** |
| ② | Script | Writes a full scene-by-scene script with dialogue and character descriptions. Runs automatically. |
| ③ | Scene Design | Converts each scene into a detailed shot specification: camera type, movement, framing, character appearance, location. |
| ④ | Media Generation | Generates all character portraits (Imagen 4 Ultra) and location reference images (Gemini) **in parallel**. **Pauses for your approval.** You can request individual regenerations. |
| ⑤ | Scene Images | Generates one image per shot, conditioned on character portraits and location references, using Gemini's image generation. |
| ⑥ | Scene Videos | Sends each scene image + prompt to Veo 3.0. Generates an 8-second 16:9 clip with embedded dialogue and ambient audio. Up to 3 scenes generated concurrently. |
| ⑦ | Assembly | Downloads all Veo clips from GCS in parallel, re-encodes for codec consistency, concatenates into a single MP4, and uploads back to GCS. |

---

## Image Quality Evaluation

Every generated image passes through a Gemini-based evaluator before being accepted. The evaluator scores on six weighted dimensions:

| Dimension | Weight | Description |
|---|---|---|
| Subject Adherence | 3 | Correct characters or environment depicted |
| Attribute Matching | 3 | Outfits, colours, features match specification |
| Spatial Accuracy | 1 | Positioning and layout match intent |
| Style Fidelity | 2 | Visual style matches (anime, cinematic, etc.) |
| Image Quality | 1 | No artefacts, sharp, well-composed |
| No Storyboards | 1 | Pure image, no text panels or comic frames |

Maximum score: **24**. Threshold to accept: **18**. Up to 3 generation attempts per image before falling back to the best available result.

---

## Session State Keys

Tools communicate exclusively through ADK session state:

| Key | Written By | Read By |
|---|---|---|
| `film_concept` | brainstorm_film_concept | write_script, design_scenes, all media generators |
| `title` | brainstorm_film_concept | video_agent, assemble_agent |
| `script` | write_script | design_scenes |
| `characters` | write_script | design_scenes, generate_character_portraits |
| `shot_list` | design_scenes | generate_location_images, generate_scene_images, generate_scene_videos |
| `character_portraits` | generate_character_portraits | generate_scene_images |
| `location_images` | generate_location_images | generate_scene_images |
| `scene_image_uris` | generate_scene_images | generate_scene_videos |
| `video_uri_map` | generate_scene_videos | assemble_film |
| `video_uris` | generate_scene_videos | assemble_film |
| `final_film_uri` | assemble_film | returned to user |

---

## 🏁 Try It — Judge's Quick Start

The agent is deployed and ready. No local setup needed.

### Option A — Live Demo (recommended)

Open the hosted UI in your browser:

```
https://adk-default-service-name-1055217599977.us-central1.run.app
```

1. Select **film_generator_agent** from the dropdown
2. Type your idea — for example:
   > *"A samurai ghost who doesn't know he's dead, anime style, bittersweet tone"*
3. The agent will ask a couple of clarifying questions about genre and mood, then generate the full film concept. **Approve it** (or ask for changes).
4. It will generate character portraits and location reference images in parallel. **Review and approve** them.
5. Sit back — scene images, Veo 3.0 video clips, and final assembly run automatically (~5–10 min).
6. The final GCS link to your MP4 is returned when done.

> **Tip:** The two approval steps are the only times you need to respond. Everything else is fully automatic.

---

### Option B — Run Locally

**Prerequisites:** Python 3.10+, a Google Cloud project with Vertex AI + GCS enabled, and a service account with `Vertex AI User` + `Storage Object Admin` roles.

```bash
# 1. Clone and install
git clone <repo-url>
cd pAssistant
pip install google-adk "moviepy>=2.0" google-cloud-storage pydantic python-dotenv

# 2. Configure environment
cp .env.example .env
# Edit .env and fill in:
#   GOOGLE_CLOUD_PROJECT=your-project-id
#   GOOGLE_CLOUD_LOCATION=us-central1
#   GCS_BUCKET_NAME=your-bucket-name

# 3. Launch the ADK web UI
adk web
```

Then open [http://localhost:8000](http://localhost:8000), select **film_generator_agent**, and follow the same steps as Option A.

---

## Setup

### Prerequisites
- Python 3.10+
- Google Cloud project with Vertex AI, GCS, and billing enabled
- Service account with roles: `Vertex AI User`, `Storage Object Admin`

### Install
```bash
pip install google-adk "moviepy>=2.0" google-cloud-storage pydantic python-dotenv
```

### Environment
Create a `.env` file in the project root:
```
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GCS_BUCKET_NAME=your-bucket-name
```

### Run
```bash
cd pAssistant
adk run film_generator_agent
```

Or open the ADK web UI:
```bash
adk web
```

---

## Findings & Learnings

### What worked well
- **Composite agent pattern** (single conversational LlmAgent + specialised sub-agents) gives the best of both worlds: natural user interaction and parallelised compute
- **Two-checkpoint approval model** significantly improves UX — users only need to intervene at concept and media stages; the rest runs silently
- **Image conditioning for Veo** (passing scene images as `gs://` URIs) dramatically improves temporal and visual consistency between scenes vs. text-only prompts
- **Semaphore-guarded concurrency** (`asyncio.Semaphore(3)`) prevents Veo quota exhaustion while still parallelising most of the work
- `asyncio.to_thread` is essential for wrapping synchronous Vertex AI calls — without it, Veo polling blocks the entire event loop

### Challenges
- **Character visual consistency** across shots required aggressive prompt engineering: each shot repeats the full canonical character description, and `_normalize_name` keys ensure portrait lookups never miss due to casing or spacing
- **Location images leaking character details** — fixed by reordering the generation prompt (ZERO characters constraint comes *before* the location description text) and explicitly forbidding character mentions in scene design's `location_description` field
- **Veo output path format** is `gs://bucket/session/title/scene_N/<hash>/sample_0.mp4` (not flat) — the video tool inspects the full nested path from the operation result
- **Python 3.10 compatibility** — `datetime.UTC` requires 3.11; replaced with `datetime.timezone.utc` throughout
- **ADK `AgentTool` import path** — must be `from google.adk.tools.agent_tool import AgentTool`, not `from google.adk.tools import AgentTool`

### Design decisions
- One shot per scene (vs. multiple camera angles) keeps the pipeline tractable and reduces total asset count
- Location images are stored as ADK artifacts (not GCS) since they are only needed as inline references during scene image generation
- Audio generation was removed in favour of Veo 3.0's native audio — Veo generates matching dialogue and ambient sound from the prompt, eliminating a full Lyria pipeline stage
