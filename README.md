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

A modern Gradio app for building, enhancing, and visualizing creative prompts for AI text-to-image models.

- **Prompt Generator:** Build detailed prompts step by step using dropdowns and custom attributes.
- **Prompt Enhancement:** Refine your prompt with Google Gemini for more creative, detailed results.
- **Image Generation:** Instantly generate images from your prompt using Stable Diffusion XL Turbo (via diffusers).
- **Modern UI:** Features a beautiful animated background and copy-to-clipboard buttons for easy workflow.

---

## Features

- 🎨 **Prompt Builder:** Select styles, subjects, moods, and more, or add your own custom attributes.
- ✨ **Prompt Enhancement:** Click 'Refine Prompt' to rewrite your prompt with Gemini (Google Generative AI).
- 🖼️ **Image Generation:** Click 'Generate Image' to visualize your prompt using SDXL Turbo.
- 📋 **Copy Buttons:** Quickly copy your prompt or enhanced prompt for use in other tools.
- 🌈 **Animated UI:** Enjoy a modern, glowing background and clean layout.

---

## Usage

1. **Select or add prompt options** in each category.
2. **View your prompt** in the Prompt Preview section.
3. **Refine your prompt** with the 'Refine Prompt' button (optional).
4. **Generate an image** from your prompt with the 'Generate Image' button.
5. **Copy** your prompt or enhanced prompt as needed.

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
   python gradio_app.py
   ```

---

## Hugging Face Spaces
- This app is ready for deployment on [Hugging Face Spaces](https://huggingface.co/spaces).
- Set your `GEMINI_API_KEY` as a secret in the Space settings.
- The app file is `gradio_app.py`.

---

## Requirements
- gradio
- google-generativeai
- diffusers
- torch
- Pillow
- pyperclip

---

## License
MIT

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference