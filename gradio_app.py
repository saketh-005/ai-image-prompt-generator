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

# --- Full Custom CSS for the Gradio App ---
full_custom_css = '''
<style>
/* General Body and Container Styling */
body {
  background: linear-gradient(135deg, #f0f7ff 0%, #e0efff 100%) !important; /* Lighter, subtle blue gradient */
  min-height: 100vh;
  font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
  color: #1a3e63 !important; /* Darker blue for text */
}

/* Force all Gradio containers to be light and consistent */
.gradio-container, .gr-block, .gr-panel, .gr-column, .gr-row, .gr-box, .gr-form, .gr-input, .gr-dropdown, .gr-group, .gr-accordion, .gr-acc-panel {
  background: #ffffff !important; /* Pure white background for blocks */
  color: #1a3e63 !important;
  border-radius: 16px !important; /* Slightly rounded corners */
  box-shadow: 0 4px 20px rgba(0, 50, 100, 0.08) !important; /* Softer, wider shadow */
  border: 1px solid #cce7ff !important; /* Light blue border */
  padding: 1.5em !important;
}

/* Labels and Headings */
.gr-label, label, .gr-dropdown label, .gr-input label, .gr-markdown, .gr-markdown *, .prose, .prose *, h1, h2, h3, h4, h5, h6 {
  color: #1a3e63 !important; /* Dark blue for labels */
  background: transparent !important;
  font-weight: 600 !important;
  font-size: 1.1em !important;
  border: none !important;
  padding: 0.2em 0.8em 0.2em 0 !important;
}

/* Inputs and Dropdowns (base styles) */
input, select, textarea, .gr-input input, .gr-input select, .gr-input textarea, .gr-dropdown select {
  border: 1.5px solid #82c2f0 !important; /* Medium blue border */
  border-radius: 10px !important;
  padding: 0.6em 1.2em !important;
  font-size: 1em !important;
  background: #f8fcff !important; /* Very light blue background for inputs */
  color: #1a3e63 !important;
  transition: border 0.3s, box-shadow 0.3s;
  box-shadow: none !important;
}
input:focus, select:focus, textarea:focus {
  border-color: #007bff !important; /* Bright blue on focus */
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25) !important; /* Light glow on focus */
  outline: none !important;
}

/* Buttons */
.gr-button, button, .copy-btn {
  background: #007bff !important; /* Primary blue for buttons */
  color: #ffffff !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 15px rgba(0, 123, 255, 0.2) !important;
  border: none !important;
  transition: background 0.3s, box-shadow 0.3s, transform 0.2s;
  padding: 0.7em 1.5em !important; /* Consistent padding */
}
.gr-button:hover, button:hover, .copy-btn:hover {
  background: #0056b3 !important; /* Darker blue on hover */
  box-shadow: 0 6px 20px rgba(0, 123, 255, 0.3) !important;
  transform: translateY(-2px); /* Slight lift on hover */
}

/* Chips (for custom attributes or similar) */
.chip {
  display: inline-block;
  background: #e0f0ff !important; /* Lighter blue for chips */
  color: #1a3e63 !important;
  border-radius: 20px !important; /* More rounded chips */
  padding: 0.4rem 1.2rem !important;
  margin: 0.3rem 0.4rem 0.3rem 0 !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
  box-shadow: 0 2px 10px rgba(0, 50, 100, 0.05) !important;
  opacity: 1;
  animation: chipIn 0.4s cubic-bezier(.4,2,.6,1) both;
  border: 1px solid #cce7ff !important; /* Border for chips */
}
@keyframes chipIn {
  0% { opacity: 0; transform: scale(0.7) translateY(10px);}
  100% { opacity: 1; transform: scale(1) translateY(0);}
}

/* Main Heading */
.main-heading {
  text-align: center;
  font-size: 3.2em;
  font-weight: 700;
  margin-top: 1em;
  margin-bottom: 0.8em;
  color: #0056b3; /* A strong blue for the main title */
  letter-spacing: 0.02em;
  text-shadow: 1px 1px 3px rgba(0,0,0,0.05);
}

/* Section Headings and Labels */
.gr-markdown h2, .gr-markdown h3, .gr-markdown h4, .gr-markdown h5, .gr-markdown h6 {
  font-size: 1.2em !important;
  font-weight: 600 !important;
  color: #1a3e63 !important;
  background: transparent !important;
  border: none !important;
  padding: 0.2em 0.8em 0.2em 0 !important;
}

/* Normal text and instructions */
.gr-markdown, .gr-markdown *, .prose, .prose *, p, li {
  font-size: 1em !important;
  color: #335d8a !important; /* Slightly lighter blue for body text */
  background: transparent !important;
  line-height: 1.6; /* Improved readability */
}

/* Make Gradio progress/status text and bar clearly visible */
.gr-progress-status, .gr-progress-bar, .gr-progress-bar * {
  color: #007bff !important;
  font-weight: 600 !important;
  font-size: 1em !important;
  background: #e0f0ff !important;
  border-radius: 8px !important;
  padding: 0.5em 1em;
}

/* Specific styling for the prompt preview textboxes */
#prompt-preview textarea, #enhanced-prompt textarea {
    background-color: #f0f7ff !important; /* Very light blue for prompt output */
    border: 1px dashed #a0d0ff !important; /* Dashed border for distinction */
    min-height: 80px; /* Ensure sufficient height */
    overflow-y: auto; /* Enable scrolling if content is long */
}

/* Image generation note */
#image-gen-note {
    color: #0056b3 !important;
    font-style: italic;
    font-size: 0.9em;
    text-align: center;
    margin-top: 1em;
    margin-bottom: 0.5em;
}

/* --- Specific Fixes for Custom Attributes and Image Container --- */

/* Ensure gr.List items and their text are styled correctly */
.gr-list-item {
    background-color: #f8fcff !important; /* Light background for list items */
    color: #1a3e63 !important; /* Dark text for list items */
    border: 1px solid #cce7ff !important;
    border-radius: 8px !important;
    margin-bottom: 0.5em !important; /* Space between items */
    padding: 0.6em 1em !important;
}

/* Targeting the input field specifically within the custom attributes section */
.gr-textbox[placeholder="Type and press Enter to add"] input {
    background-color: #f8fcff !important;
    color: #1a3e63 !important;
    border-color: #82c2f0 !important;
}

/* Styling for the gr.Image container */
.gr-image {
    background-color: #f8fcff !important; /* Very light blue background for image container */
    border: 1px solid #cce7ff !important; /* Consistent light blue border */
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 50, 100, 0.05) !important; /* Soft shadow */
    min-height: 300px; /* Ensure it's not too small when empty */
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Ensure the loading spinner is visible on the lighter background */
.gr-image .progress-spinner {
    filter: invert(1); /* Invert color of spinner to make it visible on light background */
}

/* --- Fix for Dropdown Containers (new additions for outer container) --- */

/* Target the main gr-dropdown container itself */
.gr-dropdown {
    background-color: #ffffff !important; /* Force white background for the dropdown container */
    border: 1px solid #cce7ff !important; /* Ensure light border */
    box-shadow: 0 4px 20px rgba(0, 50, 100, 0.08) !important; /* Consistent shadow */
    border-radius: 16px !important; /* Consistent border-radius */
    padding: 1.5em !important; /* Consistent padding */
}
/* Ensure labels inside dropdowns are correctly colored (might be redundant but for safety) */
.gr-dropdown label {
    color: #1a3e63 !important; /* Dark blue for dropdown labels */
}

/* Target the core dropdown input area where selected items (chips) live */
.gr-multiselect-input, .gr-dropdown-input {
    background-color: #f8fcff !important; /* Very light blue background */
    color: #1a3e63 !important; /* Dark text */
    border: 1.5px solid #82c2f0 !important;
    border-radius: 10px !important;
    padding: 0.6em 1.2em !important;
    padding-top: 0.3em !important;
    padding-bottom: 0.3em !important;
    min-height: 38px; /* Ensure a consistent height even with no selections */
}

/* Style the selected chips within the dropdowns */
.gr-multiselect-input .gradio-chip {
    background-color: #e0f0ff !important; /* Lighter blue for selected chips */
    color: #1a3e63 !important;
    border-radius: 16px !important;
    padding: 0.3rem 0.8rem !important;
    margin: 0.2rem !important;
    font-size: 0.9em !important;
    border: 1px solid #cce7ff !important;
    box-shadow: none !important;
}

/* Style the 'x' button on dropdown chips */
.gr-multiselect-input .gradio-chip button {
    background: transparent !important;
    color: #007bff !important;
    font-size: 1.1em !important;
    padding: 0 0.2em !important;
}
.gr-multiselect-input .gradio-chip button:hover {
    color: #0056b3 !important;
    background: transparent !important;
    transform: none !important;
}


/* Style the dropdown menu (the actual list of options) when it opens */
.gr-options, .gr-options > div {
    background-color: #ffffff !important;
    border: 1px solid #cce7ff !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(0, 50, 100, 0.1) !important;
}

/* Style individual items in the dropdown menu */
.gr-option {
    color: #1a3e63 !important;
    padding: 0.8em 1.2em !important;
    background-color: transparent !important;
}

.gr-option:hover {
    background-color: #e0f0ff !important;
    color: #0056b3 !important;
}

.gr-option.selected {
    background-color: #cce7ff !important;
    color: #0056b3 !important;
    font-weight: 600 !important;
}

/* Ensure the caret (down arrow) is visible */
.gr-dropdown-caret {
    color: #007bff !important;
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
      el.style.color = "#1a3e63"; /* Ensure text color after dynamic styles load */
    });
  }, 1000);
});
</script>
'''

# --- Image generation pipeline setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_repo_id = "stabilityai/sdxl-turbo"
if torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32
pipe = DiffusionPipeline.from_pretrained(model_repo_id, torch_dtype=torch_dtype)
pipe = pipe.to(device)

def generate_image(prompt_text):
    if not prompt_text or prompt_text.strip() == "":
        return None
    image = pipe(prompt=prompt_text, guidance_scale=0.0, num_inference_steps=2, width=1024, height=1024).images[0]
    return image

# Use gr.Themes.Soft() as the base theme and apply the full custom CSS
with gr.Blocks(theme=gr.themes.Soft(), css=full_custom_css) as demo:
    gr.HTML('<div class="main-heading">AI Image Prompt Generator 🎨</div>')
    with gr.Row():
        gr.Markdown("""# Create creative prompts for AI text-to-image models!
        Welcome! Build your perfect prompt step by step. Select or add options below, and copy your prompt to use in your favorite AI image tool.""")
    with gr.Row():
        with gr.Column():
            gr.Markdown("""**How to use:**
1. Choose or randomize options for each category (or add your own).
2. See your prompt update live below.
3. Click 'Copy Prompt' to use it elsewhere!""")
        with gr.Column():
            pass # Empty column for layout balance

    # Define all dropdowns
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

    gr.Markdown("---")
    gr.Markdown("#### 2. Add any extra custom attributes:")
    with gr.Row():
        custom_attr = gr.Textbox(label="Add custom attribute (anything not covered above)", placeholder="Type and press Enter to add", interactive=True)
        add_btn = gr.Button("Add Attribute")
    custom_attr_list = gr.List(label="Custom Attributes", value=[], interactive=True)

    gr.Markdown("---")
    gr.Markdown("#### 3. Your generated prompt:")
    # Prompt Preview row
    with gr.Row():
        prompt = gr.Textbox(label="Prompt Preview", interactive=False, elem_id="prompt-preview", scale=8, lines=3)
        copy_prompt_btn = gr.Button("📋", elem_classes=["copy-btn"], scale=1)
    # Enhanced Prompt row
    with gr.Row():
        # Changed label here: removed "(powered by Gemini)"
        enhanced_prompt = gr.Textbox(label="Enhanced Prompt", interactive=False, elem_id="enhanced-prompt", scale=8, lines=3)
        copy_enhanced_btn = gr.Button("📋", elem_classes=["copy-btn"], scale=1)
    refine_btn = gr.Button("Refine Prompt with Gemini 🪄", variant="secondary")

    # --- Image Generation Section ---
    gr.Markdown("---")
    gr.Markdown("#### 4. Generate image from your prompt:")
    gr.Markdown("_Note: Image generation generally takes 300-400 seconds. Please be patient!_", elem_id="image-gen-note")
    with gr.Row():
        generate_img_btn = gr.Button("Generate Image 🖼️", variant="primary")
    img_output = gr.Image(label="Generated Image", show_label=True)

    # --- Attach all event listeners AFTER all components are defined ---

    def add_custom_attribute(custom_attr_input, current_custom_attr_list):
        if custom_attr_input and [custom_attr_input] not in current_custom_attr_list:
            current_custom_attr_list = current_custom_attr_list + [[custom_attr_input]]
        return current_custom_attr_list, ""

    add_btn.click(add_custom_attribute, [custom_attr, custom_attr_list], [custom_attr_list, custom_attr])
    custom_attr.submit(add_custom_attribute, [custom_attr, custom_attr_list], [custom_attr_list, custom_attr])

    copy_prompt_btn.click(copy_to_clipboard, inputs=prompt, outputs=None, show_progress=False)
    copy_enhanced_btn.click(copy_to_clipboard, inputs=enhanced_prompt, outputs=None, show_progress=False)
    refine_btn.click(enhance_prompt_with_gemini, [prompt], [enhanced_prompt])
    generate_img_btn.click(generate_image, inputs=prompt, outputs=img_output)

    # Live update prompt preview using .change() on all relevant components
    for comp in [style, subject, mood, clothing_sel, prop_sel, pose_sel, setting_sel, scene_sel, artist_sel, color_sel, custom_attr_list]:
        comp.change(
            generate_prompt,
            [style, subject, mood, clothing_sel, prop_sel, pose_sel, setting_sel, scene_sel, artist_sel, color_sel, custom_attr_list],
            [prompt]
        )

demo.launch(share=True)