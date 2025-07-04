---
title: Ai Image Prompt Generator
emoji: 🖼
colorFrom: purple
colorTo: red
sdk: streamlit
# sdk_version: (Optional, Streamlit versions are usually handled via requirements.txt)
app_file: streamlit_app.py
pinned: false
license: mit
---

# AI Image Prompt Generator 🖼️

A modern **Streamlit** app for building, enhancing, and visualizing creative prompts for AI text-to-image models.

- **Prompt Generator:** Build detailed prompts step by step using dropdowns and custom attributes.
- **Prompt Enhancement:** Refine your prompt with Google Gemini for more creative, detailed results.
- **Image Generation:** Instantly generate images from your prompt using Stable Diffusion XL Turbo (via diffusers).
- **Modern UI:** Features a beautiful blue-themed background, prominent headings, and copy-to-clipboard buttons for easy workflow.

---

## Live Demo

Try it instantly on Hugging Face Spaces: [ai-image-prompt-generator](https://huggingface.co/spaces/saketh-005/ai-image-prompt-generator)

---

## Features

- 🎨 **Prompt Builder:** Select styles, subjects, moods, and more, or add your own custom attributes. Prompts update live as you make selections.
- ✨ **Prompt Enhancement:** Click 'Refine Prompt' to rewrite your prompt with Gemini (Google Generative AI) for more creative, detailed, and natural language.
- 🖼️ **Image Generation:** Click 'Generate Image' to visualize your prompt using SDXL Turbo (Stable Diffusion XL Turbo) directly in the app.
- 📋 **Copy Buttons:** Quickly copy your prompt or enhanced prompt for use in other tools (manual copy fallback if clipboard is unavailable).
- 🌈 **Modern UI:** Enjoy a clean, blue-gradient background, large and visually prominent headings, and a user-friendly layout.
- ⚡ **Status & Progress Feedback:** See clear status messages and progress indicators during prompt enhancement and image generation.
- 🔒 **Secure API Key Management:** Gemini API key is read from environment variables or Hugging Face secrets.

---

## Usage

1. **Select or add prompt options** in each category (Style, Subject, Mood, Clothing, Prop, Pose, Setting, Scene, Artist, Color/Lighting, or add custom attributes).
2. **View your prompt** in the Prompt Preview section (updates live as you change selections).
3. **Refine your prompt** with the 'Refine Prompt' button (optional, uses Google Gemini for enhancement).
4. **Generate an image** from your prompt with the 'Generate Image' button (uses SDXL Turbo).
5. **Copy** your prompt or enhanced prompt as needed (clipboard button or manual copy).

---

## Running Locally

1. Clone this repo and install requirements:
   ```sh
   pip install -r requirements.txt
   ```
2. Set your Gemini API key:
   ```sh
   export GEMINI_API_KEY=your-gemini-api-key
   ```
3. Run the app:
   ```sh
   streamlit run streamlit_app.py
   ```

---

## Hugging Face Spaces
- This app is ready for deployment on [Hugging Face Spaces](https://huggingface.co/spaces/saketh-005/ai-image-prompt-generator).
- Set your `GEMINI_API_KEY` as a secret in the Space settings.
- The app file is `streamlit_app.py`.

---

## Requirements
- streamlit
- google-generativeai
- diffusers
- torch
- Pillow
- pyperclip

---

## License
[MIT](LICENSE) © Saketh Jangala

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference