import streamlit as st
import os
import google.generativeai as genai
import pyperclip # Note: pyperclip works client-side on local machine, but not directly in web apps like Streamlit Cloud for security reasons. Users may need to copy manually.
import torch
from diffusers import DiffusionPipeline

# Assuming prompt_data.py exists and contains the dict
# For this example, I'll put a dummy prompt_data if the file isn't present
try:
    from prompt_data import prompt_data
except ImportError:
    st.warning("prompt_data.py not found. Using dummy data for categories. Please ensure 'prompt_data.py' is in the same directory.")
    prompt_data = {
        "DRAWING STYLES": ["Realistic", "Abstract", "Cartoon", "Anime"],
        "VISUAL STYLES": ["Oil Painting", "Watercolor", "Pixel Art", "Digital Art"],
        "SUBJECTS": ["Person", "Animal", "Landscape", "Object"],
        "EMOTIONS": ["Happy", "Sad", "Angry", "Calm"],
        "CLOTHING": ["T-Shirt", "Dress", "Suit", "Casual"],
        "PROPS": ["Sword", "Book", "Flower", "Coffee Cup"],
        "POSES": ["Standing", "Sitting", "Running", "Jumping"],
        "SETTINGS": ["Forest", "City", "Space", "Beach"],
        "SCENE": ["Sunrise", "Night", "Rainy Day", "Sunset"],
        "ARTISTS": ["Van Gogh", "Picasso", "Monet", "Leonardo da Vinci"],
        "COLORS": ["Vibrant", "Pastel", "Monochromatic", "Neon"]
    }


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
    # Attempt to initialize Vertex AI client only if project ID is not default
    if GCP_PROJECT_ID and GCP_PROJECT_ID != 'your-gcp-project-id' and 'genai_client' not in st.session_state:
        st.session_state.genai_client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_REGION)
    client = st.session_state.get('genai_client', None) # Get client from session_state or None
except Exception as e:
    client = None
    st.error(f"Vertex AI Client Initialization Error: {e}")
    st.warning("Please ensure GCP_PROJECT_ID and GCP_REGION are correct and Vertex AI API is enabled.")

# --- Prompt generator logic ---
def generate_prompt_text(style, subject, mood, clothing_sel, prop_sel, pose_sel, setting_sel, scene_sel, artist_sel, color_sel, custom_attributes):
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
    
    custom_attributes_flat = []
    if custom_attributes:
        # Streamlit's multiselect often returns strings directly if not nested
        # but if custom_attr_list was a list of lists, this handles it.
        if isinstance(custom_attributes, list) and custom_attributes and isinstance(custom_attributes[0], list):
             custom_attributes_flat = [item for sublist in custom_attributes for item in sublist]
        else:
             custom_attributes_flat = custom_attributes # Assume it's already a flat list of strings

    prompt_parts = [
        ", ".join(style), ", ".join(subject), ", ".join(mood), ", ".join(clothing_sel), ", ".join(prop_sel), ", ".join(pose_sel), ", ".join(setting_sel), ", ".join(scene_sel), ", ".join(artist_sel), ", ".join(color_sel), ", ".join(custom_attributes_flat)
    ]
    prompt = ", ".join([p for p in prompt_parts if p])
    return prompt

def enhance_prompt_with_gemini(prompt):
    if not GEMINI_API_KEY:
        st.error("Gemini API key not set. Please set GEMINI_API_KEY environment variable.")
        return "[Gemini API key not set.]"
    try:
        text_client = genai.Client(api_key=GEMINI_API_KEY)
        with st.spinner("Enhancing prompt with AI..."):
            response = text_client.models.generate_content(
                model='gemini-1.5-flash-latest',
                contents=f"Rewrite this comma-separated list as a perfect, detailed prompt for an AI image generation model: {prompt}"
            )
            return response.text.strip()
    except Exception as e:
        st.error(f"Gemini API error during prompt enhancement: {e}")
        return f"[Gemini API error: {e}]"

# --- Image generation pipeline setup ---
@st.cache_resource
def load_diffusion_pipeline():
    # Use st.cache_resource to load the model once, across all reruns
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_repo_id = "stabilityai/sdxl-turbo"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    st.info(f"Loading Diffusion Pipeline '{model_repo_id}' on {device} with dtype {torch_dtype}...")
    pipe = DiffusionPipeline.from_pretrained(model_repo_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    st.success("Diffusion Pipeline loaded!")
    return pipe

# Load the pipeline globally (will be cached)
pipe = load_diffusion_pipeline()

def generate_image_from_prompt(prompt_text):
    if not prompt_text or prompt_text.strip() == "":
        st.warning("Please generate a prompt first!")
        return None
    
    with st.spinner("Generating image (this can take 30-60 seconds locally, or longer on cloud platforms)..."):
        try:
            image = pipe(prompt=prompt_text, guidance_scale=0.0, num_inference_steps=2, width=1024, height=1024).images[0]
            st.success("Image generation complete!")
            return image
        except Exception as e:
            st.error(f"Error during image generation: {e}")
            return None

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="AI Image Prompt Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# Custom CSS for Streamlit (simplified from Gradio, focusing on blue theme where possible)
st.markdown("""
<style>
/* General body background and font */
body {
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #f0f7ff 0%, #e0efff 100%); /* Light blue gradient */
    color: #1a3e63; /* Darker blue for text */
}

/* Titles and Headers */
h1 {
    color: #0056b3; /* Strong blue for main title */
    text-align: center;
    font-size: 3.2em;
    font-weight: 700;
    margin-top: 1em;
    margin-bottom: 0.8em;
    letter-spacing: 0.02em;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.05);
}
h2, h3, h4, h5, h6 {
    color: #1a3e63; /* Dark blue for section headers */
    font-weight: 600;
}

/* Markdown text */
.stMarkdown {
    color: #335d8a; /* Slightly lighter blue for body text */
}

/* Input fields, select boxes, text areas */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stMultiSelect > div:first-child > div, /* Multi-select input area */
.stSelectbox > div:first-child > div { /* Selectbox input area */
    background-color: #f8fcff !important; /* Very light blue */
    border: 1.5px solid #82c2f0 !important; /* Medium blue border */
    border-radius: 10px !important;
    color: #1a3e63 !important;
}
.stTextInput > label, .stTextArea > label, .stMultiSelect > label, .stSelectbox > label {
    color: #1a3e63 !important; /* Dark blue for labels */
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    background-color: #007bff; /* Primary blue */
    color: white;
    border-radius: 12px;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(0, 123, 255, 0.2);
    border: none;
    transition: background-color 0.3s, box-shadow 0.3s, transform 0.2s;
    padding: 0.7em 1.5em;
}
.stButton > button:hover {
    background-color: #0056b3; /* Darker blue on hover */
    box-shadow: 0 6px 20px rgba(0, 123, 255, 0.3);
    transform: translateY(-2px);
}

/* Multi-select selected items (chips) */
.stMultiSelect div[data-baseweb="tag"] {
    background-color: #e0f0ff !important; /* Lighter blue for chips */
    color: #1a3e63 !important;
    border-radius: 16px !important;
    border: 1px solid #cce7ff !important;
}
.stMultiSelect div[data-baseweb="tag"] span {
    color: #1a3e63 !important; /* Text inside chip */
}
.stMultiSelect div[data-baseweb="tag"] svg {
    fill: #007bff !important; /* 'x' icon color */
}

/* Text areas for prompt preview */
.stTextArea[aria-label="Prompt Preview"] textarea,
.stTextArea[aria-label="Enhanced Prompt"] textarea {
    background-color: #f0f7ff !important; /* Very light blue for prompt output */
    border: 1px dashed #a0d0ff !important; /* Dashed border for distinction */
    min-height: 80px;
    overflow-y: auto;
}

/* Info/Warning/Success messages and Spinners */
/* The specific class might change with Streamlit versions, but this targets the common alert container */
.st-emotion-cache-p5m87v { /* This might need adjustment based on Streamlit version */
    background-color: #e0f0ff; /* Light blue for info/spinner backgrounds */
    border-radius: 8px;
    border: 1px solid #cce7ff;
}
.st-emotion-cache-p5m87v > div > p {
    color: #1a3e63 !important;
}

/* Image container */
.stImage {
    background-color: #f8fcff; /* Very light blue background for image container */
    border: 1px solid #cce7ff; /* Consistent light blue border */
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0, 50, 100, 0.05);
    min-height: 300px; /* Ensure it's not too small when empty */
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1em; /* Add padding inside the image container */
}

/* Copy button specific style for icon */
.copy-btn {
    font-size: 1.2em !important;
    line-height: 1;
    padding: 0.5em 0.8em !important; /* Smaller padding for icon button */
    display: flex;
    align-items: center;
    justify-content: center;
}

</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-heading">AI Image Prompt Generator 🎨</div>', unsafe_allow_html=True)
st.markdown("""
# Create creative prompts for AI text-to-image models!

Welcome! Build your perfect prompt step by step. Select or add options below, and copy your prompt to use in your favorite AI image tool.
""")

col1_inst, col2_inst = st.columns(2)
with col1_inst:
    st.markdown("""
    **How to use:**
    1. Choose or randomize options for each category (or add your own).
    2. See your prompt update live below.
    3. Copy your prompt to use it elsewhere! (Manual copy needed for now, or use the clipboard icon buttons)
    """)
with col2_inst:
    pass # Empty column for layout balance

# Initialize session state variables if not already present
if 'custom_attributes' not in st.session_state:
    st.session_state.custom_attributes = []
if 'generated_prompt' not in st.session_state:
    st.session_state.generated_prompt = ""
if 'enhanced_prompt' not in st.session_state:
    st.session_state.enhanced_prompt = ""
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
if 'copy_message' not in st.session_state:
    st.session_state.copy_message = ""
# Store current selections to detect changes and trigger update_prompt
if 'current_selections' not in st.session_state:
    st.session_state.current_selections = {}

# Function to update prompt on any selection change
def update_prompt():
    current_selections = {
        "style": st.session_state.get('style_select', []),
        "subject": st.session_state.get('subject_select', []),
        "mood": st.session_state.get('mood_select', []),
        "clothing": st.session_state.get('clothing_select', []),
        "prop": st.session_state.get('prop_select', []),
        "pose": st.session_state.get('pose_select', []),
        "setting": st.session_state.get('setting_select', []),
        "scene": st.session_state.get('scene_select', []),
        "artist": st.session_state.get('artist_select', []),
        "color": st.session_state.get('color_select', []),
        "custom_attributes": st.session_state.custom_attributes # Custom attributes also trigger
    }

    # Only update if selections have actually changed
    if current_selections != st.session_state.current_selections:
        st.session_state.current_selections = current_selections.copy() # Store a copy

        st.session_state.generated_prompt = generate_prompt_text(
            current_selections["style"], current_selections["subject"], current_selections["mood"],
            current_selections["clothing"], current_selections["prop"], current_selections["pose"],
            current_selections["setting"], current_selections["scene"], current_selections["artist"],
            current_selections["color"], current_selections["custom_attributes"]
        )
        # Clear enhanced prompt if base prompt changes
        st.session_state.enhanced_prompt = ""
        st.session_state.generated_image = None # Clear generated image if prompt changes

# Define selection widgets
st.markdown("---")
st.markdown("#### 1. Select prompt attributes:")

cols_sel_1 = st.columns(3)
with cols_sel_1[0]:
    st.multiselect("Style", options=styles, key="style_select", on_change=update_prompt)
with cols_sel_1[1]:
    st.multiselect("Subject", options=subjects, key="subject_select", on_change=update_prompt)
with cols_sel_1[2]:
    st.multiselect("Mood/Emotion", options=moods, key="mood_select", on_change=update_prompt)

cols_sel_2 = st.columns(3)
with cols_sel_2[0]:
    st.multiselect("Clothing", options=clothing, key="clothing_select", on_change=update_prompt)
with cols_sel_2[1]:
    st.multiselect("Prop", options=props, key="prop_select", on_change=update_prompt)
with cols_sel_2[2]:
    st.multiselect("Pose", options=poses, key="pose_select", on_change=update_prompt)

cols_sel_3 = st.columns(3)
with cols_sel_3[0]:
    st.multiselect("Setting", options=settings, key="setting_select", on_change=update_prompt)
with cols_sel_3[1]:
    st.multiselect("Scene", options=scenes, key="scene_select", on_change=update_prompt)
with cols_sel_3[2]:
    st.multiselect("Artist", options=artists, key="artist_select", on_change=update_prompt)

st.multiselect("Color/Lighting", options=colors, key="color_select", on_change=update_prompt)


st.markdown("---")
st.markdown("#### 2. Add any extra custom attributes:")
cols_custom = st.columns([0.7, 0.3])
with cols_custom[0]:
    custom_attr_input = st.text_input("Add custom attribute (type and press Enter or click 'Add')", placeholder="Type and press Enter to add", key="custom_attr_input")
with cols_custom[1]:
    st.markdown(" ") # Visual spacing for button alignment
    add_attr_button = st.button("Add Attribute", key="add_attr_button")

# Logic to add custom attribute
if add_attr_button and custom_attr_input:
    if custom_attr_input not in st.session_state.custom_attributes:
        st.session_state.custom_attributes.append(custom_attr_input)
    st.session_state.custom_attr_input = "" # Clear input field after adding
    update_prompt() # Update prompt after adding custom attribute
    st.rerun() # Rerun to display updated list immediately

if st.session_state.custom_attributes:
    st.markdown("##### Custom Attributes:")
    # Display custom attributes and provide remove buttons
    for i, attr in enumerate(st.session_state.custom_attributes):
        attr_col, remove_col = st.columns([0.9, 0.1])
        with attr_col:
            st.markdown(f"- `{attr}`")
        with remove_col:
            if st.button("x", key=f"remove_attr_{i}"):
                st.session_state.custom_attributes.pop(i)
                update_prompt() # Update prompt after removing custom attribute
                st.rerun() # Rerun to remove item immediately


st.markdown("---")
st.markdown("#### 3. Your generated prompt:")

cols_prompt = st.columns([0.85, 0.15])
with cols_prompt[0]:
    st.text_area("Prompt Preview", value=st.session_state.generated_prompt, height=100, key="prompt_preview_output", help="This is your raw generated prompt.", disabled=True)
with cols_prompt[1]:
    st.markdown("<div style='height: 1.8em;'></div>", unsafe_allow_html=True) # Adjust vertical alignment
    # Check if pyperclip is available for copy button
    if 'pyperclip_available' not in st.session_state:
        try:
            pyperclip.copy("test")
            st.session_state.pyperclip_available = True
        except pyperclip.PyperclipException:
            st.session_state.pyperclip_available = False

    if st.session_state.pyperclip_available:
        if st.button("📋", key="copy_prompt_btn", help="Copy Prompt to Clipboard", type="secondary"):
            pyperclip.copy(st.session_state.generated_prompt)
            st.toast("Prompt copied to clipboard!")
    else:
        st.markdown("*(Manual copy)*", help="Copy button disabled. Please copy manually.")


cols_enhanced = st.columns([0.85, 0.15])
with cols_enhanced[0]:
    st.text_area("Enhanced Prompt", value=st.session_state.enhanced_prompt, height=100, key="enhanced_prompt_output", help="This prompt is refined by an AI model.", disabled=True)
with cols_enhanced[1]:
    st.markdown("<div style='height: 1.8em;'></div>", unsafe_allow_html=True) # Adjust vertical alignment
    if st.session_state.pyperclip_available:
        if st.button("📋", key="copy_enhanced_btn", help="Copy Enhanced Prompt to Clipboard", type="secondary"):
            pyperclip.copy(st.session_state.enhanced_prompt)
            st.toast("Enhanced prompt copied to clipboard!")
    else:
        st.markdown("*(Manual copy)*", help="Copy button disabled. Please copy manually.")


# Refine Prompt Button
if st.button("Refine Prompt 🪄", key="refine_btn"):
    if st.session_state.generated_prompt:
        with st.spinner("Refining prompt with Gemini..."):
            st.session_state.enhanced_prompt = enhance_prompt_with_gemini(st.session_state.generated_prompt)
        st.success("Prompt refined!")
    else:
        st.warning("Please generate a base prompt first before refining.")


st.markdown("---")
st.markdown("#### 4. Generate image from your prompt:")
st.markdown("_Note: Image generation generally takes 30-60 seconds locally, or longer on Hugging Face Spaces. Please be patient!_")

if st.button("Generate Image 🖼️", key="generate_img_btn", type="primary"):
    # Always use the current generated_prompt from session state
    if st.session_state.generated_prompt:
        st.session_state.generated_image = generate_image_from_prompt(st.session_state.generated_prompt)
    else:
        st.warning("Please generate a prompt first to create an image!")

if st.session_state.generated_image:
    st.image(st.session_state.generated_image, caption="Generated Image", use_column_width=True)

# Call update_prompt initially to set the default prompt if no selections are made yet
update_prompt()