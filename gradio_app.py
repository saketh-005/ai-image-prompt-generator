import gradio as gr
from prompt_data import prompt_data
import os
import google.generativeai as genai
from io import BytesIO
import pyperclip
from google import genai
from google.genai import types
import torch
from diffusers import DiffusionPipeline

# Helper to get combined styles
styles = prompt_data.get("DRAWING STYLES", []) + prompt_data.get("VISUAL STYLES", [])
subjects = prompt_data.get("SUBJECTS", [])
moods = prompt_data.get("EMOTIONS", [])
clothing = prompt_data.get("CLOTHING", [])
props = prompt_data.get("PROPS", [])
poses = prompt_data.get("POSES", [])
settings = prompt_data.get("SETTINGS", [])
scenes = prompt_data.get("SCENE", [])
artists = prompt_data.get("ARTISTS", [])
colors = prompt_data.get("COLORS", [])

# Configure Gemini API key (for prompt enhancement)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- NEW: Vertex AI Configuration for Imagen ---
# REPLACE THESE WITH YOUR ACTUAL GOOGLE CLOUD PROJECT ID AND REGION
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'your-gcp-project-id') # e.g., 'my-ai-project-12345'
GCP_REGION = os.getenv('GCP_REGION', 'us-central1') # e.g., 'us-central1'

# Initialize google-genai client for Vertex AI
try:
    client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_REGION)
except Exception as e:
    client = None
    print(f"Vertex AI Client Initialization Error: {e}")
    print("Please ensure GCP_PROJECT_ID and GCP_REGION are correct and Vertex AI API is enabled.")

# --- Prompt generator logic ---
def generate_prompt(style, subject, mood, clothing_sel, prop_sel, pose_sel, setting_sel, scene_sel, artist_sel, color_sel, custom_attributes):
    # Ensure all are lists, not None
    def ensure_list(x):
        return x if isinstance(x, list) else ([] if x is None else [x])
    style = ensure_list(style)
    subject = ensure_list(subject)
    mood = ensure_list(mood)
    clothing_sel = ensure_list(clothing_sel)
    prop_sel = ensure_list(prop_sel)
    pose_sel = ensure_list(pose_sel)
    setting_sel = ensure_list(setting_sel)
    scene_sel = ensure_list(scene_sel)
    artist_sel = ensure_list(artist_sel)
    color_sel = ensure_list(color_sel)
    # Handle custom_attributes
    if not custom_attributes:
        custom_attributes_flat = []
    elif isinstance(custom_attributes[0], list):
        custom_attributes_flat = [item for sublist in custom_attributes for item in sublist]
    else:
        custom_attributes_flat = custom_attributes
    prompt_parts = [
        ", ".join(style), ", ".join(subject), ", ".join(mood), ", ".join(clothing_sel), ", ".join(prop_sel), ", ".join(pose_sel), ", ".join(setting_sel), ", ".join(scene_sel), ", ".join(artist_sel), ", ".join(color_sel), ", ".join(custom_attributes_flat)
    ]
    prompt = ", ".join([p for p in prompt_parts if p])
    return prompt

def enhance_prompt_with_gemini(prompt):
    if not GEMINI_API_KEY:
        return "[Gemini API key not set. Please set GEMINI_API_KEY environment variable.]"
    try:
        # Use the Gemini 1.5 Flash model for lower latency/quota
        text_client = genai.Client(api_key=GEMINI_API_KEY)
        response = text_client.models.generate_content(
            model='gemini-1.5-flash-latest',
            contents=f"Rewrite this comma-separated list as a perfect, detailed prompt for an AI image generation model: {prompt}"
        )
        return response.text.strip()
    except Exception as e:
        return f"[Gemini API error during prompt enhancement: {e}]"

def copy_to_clipboard(text):
    try:
        pyperclip.copy(text)
        return 'Copied!'
    except Exception as e:
        return f'Copy failed: {str(e)}'

# --- Custom HTML/JS/CSS for animated background and chips ---
custom_html = '''
<style>
body {
  background: linear-gradient(135deg, #e0f3ff 0%, #b3e0ff 100%) !important;
  min-height: 100vh;
  font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
  color: #22577a !important;
}
/* Force all Gradio containers to be light */
.gradio-container, .gr-block, .gr-panel, .gr-column, .gr-row, .gr-box, .gr-form, .gr-input, .gr-dropdown, .gr-group, .gr-accordion, .gr-acc-panel {
  background: #fff !important;
  color: #22577a !important;
  border-radius: 18px !important;
  box-shadow: 0 2px 16px rgba(67,164,224,0.10) !important;
  border: 1.5px solid #e0f3ff !important;
  padding: 1.2em !important;
}
/* Labels and headings */
.gr-label, label, .gr-dropdown label, .gr-input label, .gr-markdown, .gr-markdown *, .prose, .prose *, h1, h2, h3, h4, h5, h6 {
  color: #22577a !important;
  background: transparent !important;
  font-weight: 600 !important;
  font-size: 1.1em !important;
  border: none !important;
  padding: 0.2em 0.8em 0.2em 0 !important;
}
/* Inputs and dropdowns */
input, select, textarea, .gr-input input, .gr-input select, .gr-input textarea, .gr-dropdown select {
  border: 1.5px solid #43a4e0 !important;
  border-radius: 8px !important;
  padding: 0.5em 1em !important;
  font-size: 1em !important;
  background: #e0f3ff !important;
  color: #22577a !important;
  transition: border 0.2s;
  box-shadow: none !important;
}
input:focus, select:focus, textarea:focus {
  border-color: #2563eb !important;
  outline: none !important;
}
/* Buttons */
.gr-button, button, .copy-btn {
  background: #2563eb !important;
  color: #fff !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 8px rgba(67,164,224,0.10) !important;
  border: none !important;
  transition: background 0.2s, color 0.2s, box-shadow 0.2s;
}
.gr-button:hover, button:hover, .copy-btn:hover {
  background: #43a4e0 !important;
  color: #fff !important;
  box-shadow: 0 4px 16px rgba(67,164,224,0.15) !important;
}
/* Chips */
.chip {
  display: inline-block;
  background: #e0f3ff !important;
  color: #22577a !important;
  border-radius: 16px !important;
  padding: 0.3rem 1rem !important;
  margin: 0.2rem 0.3rem 0.2rem 0 !important;
  font-size: 1rem !important;
  font-weight: 500 !important;
  box-shadow: 0 1px 8px rgba(67,164,224,0.10) !important;
  opacity: 1;
  animation: chipIn 0.4s cubic-bezier(.4,2,.6,1) both;
}
@keyframes chipIn {
  0% { opacity: 0; transform: scale(0.7) translateY(10px);}
  100% { opacity: 1; transform: scale(1) translateY(0);}
}
.main-heading {
  text-align: center;
  font-size: 3.8em;
  font-weight: 700;
  margin-top: 1.2em;
  margin-bottom: 0.5em;
  color: #22577a;
  letter-spacing: 0.01em;
}
/* Section headings and labels */
.gr-markdown h2, .gr-markdown h3, .gr-markdown h4, .gr-markdown h5, .gr-markdown h6,
.gr-label, label, .gr-dropdown label, .gr-input label {
  font-size: 1.35em !important;
  font-weight: 600 !important;
  color: #22577a !important;
  background: transparent !important;
  border: none !important;
  padding: 0.2em 0.8em 0.2em 0 !important;
}
/* Normal text and instructions */
.gr-markdown, .gr-markdown *, .prose, .prose *, p, li {
  font-size: 1em !important;
  color: #22577a !important;
  background: transparent !important;
}
/* Make Gradio progress/status text and bar clearly visible */
.gr-progress-status, .gr-progress-bar, .gr-progress-bar * {
  color: #22577a !important;
  font-weight: 700 !important;
  font-size: 1.1em !important;
  background: #e0f3ff !important;
  border-radius: 8px !important;
}
</style>
<div class="glow-overlay"></div>
<script>
document.addEventListener("DOMContentLoaded", function() {
  const overlay = document.querySelector('.glow-overlay');
  if (!overlay) return;
  document.body.addEventListener("mousemove", (event) => {
    overlay.style.setProperty("--glow-x", `${event.clientX}px`);
    overlay.style.setProperty("--glow-y", `${event.clientY}px`);
    overlay.style.setProperty("--glow-opacity", "1");
  });
  document.body.addEventListener("mouseleave", () => {
    overlay.style.setProperty("--glow-opacity", "0");
  });
  setTimeout(() => {
    document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, strong, em, .gr-markdown, .prose, .gr-block, .gradio-container').forEach(el => {
      el.style.color = "#22577a";
    });
  }, 1000);
});
</script>
'''

# Add chip CSS for button styling
chip_css = '''
<style>
[id^=chip-btn-] {
  display: inline-block !important;
  background: rgba(255,255,255,0.7) !important;
  border-radius: 16px !important;
  color: #22577a !important;
  font-size: 1rem !important;
  font-weight: 500 !important;
  margin: 0.2rem 0.3rem 0.2rem 0 !important;
  padding: 0.3rem 1rem !important;
  border: none !important;
  box-shadow: 0 1px 8px rgba(67,164,224,0.10) !important;
  cursor: pointer !important;
  transition: background 0.3s, box-shadow 0.3s, transform 0.2s, opacity 0.3s;
}
[id^=chip-btn-]:hover {
  background: #e0f3ff !important;
  color: #43a4e0 !important;
  transform: scale(1.05);
}
</style>
'''

# Helper to return gr.Button.update objects for chip buttons
MAX_CHIPS = 10

def get_chip_btns_updates(custom_attributes):
    updates = []
    for i in range(MAX_CHIPS):
        if i < len(custom_attributes):
            updates.append(gr.Button.update(value=custom_attributes[i] + "  ×", visible=True))
        else:
            updates.append(gr.Button.update(value="", visible=False))
    return updates

# --- Image generation pipeline setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_repo_id = "stabilityai/sdxl-turbo"
if torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32
pipe = DiffusionPipeline.from_pretrained(model_repo_id, torch_dtype=torch_dtype)
pipe = pipe.to(device)

def generate_image(prompt):
    if not prompt or prompt.strip() == "":
        return None
    image = pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=2, width=1024, height=1024).images[0]
    return image

with gr.Blocks(theme="default", css="body { font-family: 'Segoe UI', 'Roboto', sans-serif; }") as demo:
    gr.HTML(custom_html)
    gr.HTML('<div class="main-heading">AI Image Prompt Generator 🎨</div>')
    with gr.Row():
        gr.Markdown("""#Create creative prompts for AI text-to-image models!\n\nWelcome! Build your perfect prompt step by step. Select or add options below, and copy your prompt to use in your favorite AI image tool.""")
    with gr.Row():
        with gr.Column():
            gr.Markdown("""**How to use:**\n1. Choose or randomize options for each category (or add your own).\n2. See your prompt update live below.\n3. Click 'Copy Prompt' to use it elsewhere!""")
        with gr.Column():
            pass
    with gr.Row():
        style = gr.Dropdown(choices=styles, multiselect=True, label="Style")
        subject = gr.Dropdown(choices=subjects, multiselect=True, label="Subject")
        mood = gr.Dropdown(choices=moods, multiselect=True, label="Mood/Emotion")
    with gr.Row():
        clothing_sel = gr.Dropdown(choices=clothing, multiselect=True, label="Clothing")
        prop_sel = gr.Dropdown(choices=props, multiselect=True, label="Prop")
        pose_sel = gr.Dropdown(choices=poses, multiselect=True, label="Pose")
    with gr.Row():
        setting_sel = gr.Dropdown(choices=settings, multiselect=True, label="Setting")
        scene_sel = gr.Dropdown(choices=scenes, multiselect=True, label="Scene")
        artist_sel = gr.Dropdown(choices=artists, multiselect=True, label="Artist")
    color_sel = gr.Dropdown(choices=colors, multiselect=True, label="Color/Lighting")

    gr.Markdown("---\n#### 2. Add any extra custom attributes:")
    with gr.Row():
        custom_attr = gr.Textbox(label="Add custom attribute (anything not covered above)", placeholder="Type and press Enter to add", interactive=True)
        custom_attributes = gr.State([])
        add_btn = gr.Button("Add Attribute")
    custom_attr_list = gr.List(label="Custom Attributes", value=[], interactive=True)

    def add_custom_attribute(custom_attr, custom_attr_list):
        # Ensure each entry is a list of one string, and avoid duplicates
        if custom_attr and [custom_attr] not in custom_attr_list:
            custom_attr_list = custom_attr_list + [[custom_attr]]
        return custom_attr_list, ""

    add_btn.click(add_custom_attribute, [custom_attr, custom_attr_list], [custom_attr_list, custom_attr])
    custom_attr.submit(add_custom_attribute, [custom_attr, custom_attr_list], [custom_attr_list, custom_attr])
    custom_attr_list.change(lambda l: l, [custom_attr_list], [custom_attr_list])

    gr.Markdown("---\n#### 3. Your generated prompt:")
    # Prompt Preview row
    with gr.Row():
        prompt = gr.Textbox(label="Prompt Preview", interactive=False, elem_id="prompt-preview", scale=8)
        copy_prompt_btn = gr.Button("📋", elem_classes=["copy-btn"], scale=1)
        copy_prompt_btn.click(copy_to_clipboard, inputs=prompt, outputs=None, show_progress=False)
    # Enhanced Prompt row
    with gr.Row():
        enhanced_prompt = gr.Textbox(label="Enhanced Prompt", interactive=False, elem_id="enhanced-prompt", scale=8)
        copy_enhanced_btn = gr.Button("📋", elem_classes=["copy-btn"], scale=1)
        copy_enhanced_btn.click(copy_to_clipboard, inputs=enhanced_prompt, outputs=None, show_progress=False)
    refine_btn = gr.Button("Refine Prompt 🪄")
    refine_btn.click(enhance_prompt_with_gemini, [prompt], [enhanced_prompt])

    # --- Image Generation Section ---
    gr.Markdown("---\n#### 4. Generate image from your prompt:")
    gr.Markdown("_Note: Image generation generally takes 300-400 seconds. Please be patient!_", elem_id="image-gen-note")
    with gr.Row():
        generate_img_btn = gr.Button("Generate Image 🖼️", variant="primary")
    img_output = gr.Image(label="Generated Image", show_label=True)
    generate_img_btn.click(generate_image, inputs=prompt, outputs=img_output)

    # Live update prompt preview using .change() on all relevant components
    for comp in [style, subject, mood, clothing_sel, prop_sel, pose_sel, setting_sel, scene_sel, artist_sel, color_sel, custom_attr_list]:
        comp.change(
            generate_prompt,
            [style, subject, mood, clothing_sel, prop_sel, pose_sel, setting_sel, scene_sel, artist_sel, color_sel, custom_attr_list],
            [prompt]
        )

demo.launch(share=True)