# KryinCaption Turbo — High-Speed Video Captioning Agent

An evaluation-optimized video captioning system built for extreme speed and high quality. Uses streaming-first cv2 frame extraction, a two-stage scene intelligence pipeline, and a Gemma-based critic model with automatic fallback.

---

## ⚙️ Quick Start

### Build the Docker Image
```bash
docker build -t kryin-captioning .
```

---

### Option 1 — Configure via `.env` file (Recommended)

1. Create a file named `.env` in the root directory.
2. Add **any one** of your working API keys:
   ```ini
   FIREWORKS_API_KEY=your_fireworks_key_here
   # OR
   GOOGLE_API_KEY=your_google_key_here
   # OR
   OPENROUTER_API_KEY=your_openrouter_key_here
   ```
3. Run the container based on your environment:

#### Windows Command Prompt (CMD)
```cmd
docker run --rm --env-file .env -v "%cd%/input:/input" -v "%cd%/output:/output" kryin-captioning
```

#### Windows PowerShell
```powershell
docker run --rm --env-file .env -v "${PWD}/input:/input" -v "${PWD}/output:/output" kryin-captioning
```

#### Linux / macOS
```bash
docker run --rm --env-file .env -v "$(pwd)/input:/input" -v "$(pwd)/output:/output" kryin-captioning
```

---

### Option 2 — Pass API Key directly in Command (Zero Config)

Choose your provider and copy the command for your specific terminal.

#### 1. Google Gemini

*   **Windows Command Prompt (CMD)**:
    ```cmd
    docker run --rm -e GOOGLE_API_KEY=your_google_key_here -v "%cd%/input:/input" -v "%cd%/output:/output" kryin-captioning
    ```
*   **Windows PowerShell**:
    ```powershell
    docker run --rm -e GOOGLE_API_KEY="your_google_key_here" -v "${PWD}/input:/input" -v "${PWD}/output:/output" kryin-captioning
    ```
*   **Linux / macOS**:
    ```bash
    docker run --rm -e GOOGLE_API_KEY="your_google_key_here" -v "$(pwd)/input:/input" -v "$(pwd)/output:/output" kryin-captioning
    ```

#### 2. Fireworks AI

*   **Windows Command Prompt (CMD)**:
    ```cmd
    docker run --rm -e FIREWORKS_API_KEY=your_fireworks_key_here -v "%cd%/input:/input" -v "%cd%/output:/output" kryin-captioning
    ```
*   **Windows PowerShell**:
    ```powershell
    docker run --rm -e FIREWORKS_API_KEY="your_fireworks_key_here" -v "${PWD}/input:/input" -v "${PWD}/output:/output" kryin-captioning
    ```
*   **Linux / macOS**:
    ```bash
    docker run --rm -e FIREWORKS_API_KEY="your_fireworks_key_here" -v "$(pwd)/input:/input" -v "$(pwd)/output:/output" kryin-captioning
    ```

#### 3. OpenRouter

*   **Windows Command Prompt (CMD)**:
    ```cmd
    docker run --rm -e OPENROUTER_API_KEY=your_openrouter_key_here -v "%cd%/input:/input" -v "%cd%/output:/output" kryin-captioning
    ```
*   **Windows PowerShell**:
    ```powershell
    docker run --rm -e OPENROUTER_API_KEY="your_openrouter_key_here" -v "${PWD}/input:/input" -v "${PWD}/output:/output" kryin-captioning
    ```
*   **Linux / macOS**:
    ```bash
    docker run --rm -e OPENROUTER_API_KEY="your_openrouter_key_here" -v "$(pwd)/input:/input" -v "$(pwd)/output:/output" kryin-captioning
    ```

---

## 🎭 Main Features

*   **High-Speed cv2 Extraction**: Directly streams and decodes 24 frames from the video URL, avoiding full file downloads when possible.
*   **Two-Stage Scene Processing**: First analyzes overall mood and irony candidate details, and then generates all 4 caption styles (Formal, Sarcastic, Tech Humorous, Non-Tech Humorous) in parallel to ensure high semantic accuracy.
*   **Gemma critic & Targeted Rewrites**: Reviews caption quality using Gemma/Gemini models and dynamically refines any style that scores below the target threshold.
*   **Robust Multi-Provider Fallback**: If a provider fails or encounters rate limits, automatically falls back to secondary endpoints.
