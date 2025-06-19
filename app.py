import streamlit as st
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from clarifai.client.model import Model # Import Clarifai Model client

# Environment variables
CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not CLARIFAI_PAT:
    st.error("Please set CLARIFAI_PAT environment variable.")
    st.stop()

if not SERPER_API_KEY:
    st.error("Please set SERPER_API_KEY environment variable.")
    st.stop()

# Configure Clarifai LLM for text generation
clarifai_llm = LLM(
    model="openai/gcp/generate/models/gemini-2_5-pro", # This is for your text generation LLM
    api_key=CLARIFAI_PAT,
    base_url="https://api.clarifai.com/v2/ext/openai/v1"
)

# Initialize tools
search_tool = SerperDevTool()

# --- NEW: Custom Clarifai Image Generation Tool ---
class ClarifaiImageGenTool:
    """
    A custom tool to integrate Clarifai's image generation models with CrewAI.
    """
    name: str = "Clarifai_Image_Generator"
    description: str = "Generates an image using a specified Clarifai text-to-image model based on a given text prompt."
    
    def __init__(self, model_url: str, clarifai_pat: str):
        self.model_url = model_url
        self.clarifai_pat = clarifai_pat
        self.clarifai_model = Model(url=self.model_url, pat=self.clarifai_pat)

    def _run(self, prompt: str) -> str:
        """
        Generates an image using the Clarifai model and returns its URL.
        """
        try:
            inference_params = dict(
                quality="standard", # or "hd" if model supports
                size='1024x1024'    # adjust based on model capability and desired output
            )
            
            model_prediction = self.clarifai_model.predict_by_bytes(
                prompt.encode(),
                input_type="text",
                inference_params=inference_params
            )
            
            # Clarifai's API returns a URL for the generated image.
            # The exact path might vary slightly based on the model's output structure.
            # Inspect the model_prediction object to confirm the exact path.
            # Common path is model_prediction.outputs[0].data.image.url
            if model_prediction.outputs and model_prediction.outputs[0].data.image:
                image_url = model_prediction.outputs[0].data.image.url
                return image_url
            else:
                return f"Error: No image URL found in Clarifai model response for prompt: {prompt}"
        except Exception as e:
            st.error(f"Error generating image with Clarifai: {e}")
            return f"Error generating image with Clarifai: {e}. No image available."

# Instantiate the Clarifai Image Generation Tool
# Choose your desired Clarifai-hosted image generation model URL
# Examples:
# IMAGEN_MODEL_URL = "https://clarifai.com/gcp/generate/models/imagen-2"
STABLE_DIFFUSION_XL_MODEL_URL = "https://clarifai.com/stability-ai/stable-diffusion-2/models/stable-diffusion-xl"
# DALL_E_3_CLARIFAI_WRAPPER_URL = "https://clarifai.com/openai/dall-e/models/dall-e-3" # If you prefer DALL-E via Clarifai

clarifai_image_gen_tool = ClarifaiImageGenTool(
    model_url=STABLE_DIFFUSION_XL_MODEL_URL, # Or IMAGEN_MODEL_URL, etc.
    clarifai_pat=CLARIFAI_PAT
)
# ----------------------------------------------


# Define Agents
social_media_researcher = Agent(
    role="Social Media Trend Analyst",
    goal="Identify trending topics, relevant hashtags, and platform-specific content styles for social media posts, including visual trends and detailed image concepts.",
    backstory="""You are a sharp social media analyst with a keen eye for what's hot and what resonates
    with online audiences. You excel at finding relevant data, understanding platform nuances, and identifying
    the types of visuals that perform best for different content and platforms. Your output includes a highly detailed, single-sentence image generation prompt.""",
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,
    llm=clarifai_llm
)

image_creator = Agent(
    role="AI Image Creator for Social Media",
    goal="Generate a visually compelling image that perfectly complements the social media post based on a detailed prompt provided by the researcher.",
    backstory="""You are an expert AI artist, skilled in using Clarifai's powerful text-to-image models to create stunning and relevant visuals.
    You take specific, well-crafted text prompts and transform them into high-quality images suitable for social media engagement.""",
    tools=[clarifai_image_gen_tool], # Use the custom Clarifai tool
    verbose=True,
    allow_delegation=False,
    llm=clarifai_llm
)

social_media_writer = Agent(
    role="Creative Social Media Strategist & Writer",
    goal="Craft compelling, concise, and platform-optimized social media posts with engaging copy, emojis, hashtags, and integrated visuals.",
    backstory="""You are a master of digital storytelling, able to condense complex ideas into impactful
    social media snippets. You know how to capture attention and drive engagement across different platforms,
    seamlessly integrating text with powerful imagery. Your output is always markdown-formatted and ready to use.""",
    verbose=True,
    allow_delegation=True,
    llm=clarifai_llm
)

def create_social_media_tasks(topic, platform):
    """Create research, image generation, and writing tasks for social media content."""
    research_task = Task(
        description=f"""Conduct a comprehensive investigation into '{topic}' specifically for social media.
        Identify key facts, interesting angles, trending discussions, and popular hashtags.
        Also, analyze common content formats, tone, and *visual styles* used on '{platform}' for similar topics.
        Crucially, generate a *detailed, single-sentence image generation prompt* that the image creator can directly use to make an impactful image for this topic and platform.
        Example image prompt: "A futuristic city skyline at sunset, with holographic advertisements and a diverse group of people walking, in a vibrant, photorealistic style, suitable for a LinkedIn post about urban innovation."
        """,
        expected_output=f"""A concise bullet-point summary of key information, trending hashtags (at least 5),
        suggested content styles/tones, AND a single, detailed image generation prompt (string) for the image creator,
        formatted as 'IMAGE_PROMPT: "Your detailed image prompt here."' at the end of the output.
        Ensure the IMAGE_PROMPT is the last line of your output and easy to parse.""",
        agent=social_media_researcher
    )

    image_generation_task = Task(
        description=f"""Extract the 'IMAGE_PROMPT' from the researcher's output and use it to generate an image using the Clarifai image generation tool.
        The prompt will be specific and tailored for social media visual appeal on '{platform}'.
        Your task is to simply execute the image generation tool with this prompt.
        """,
        expected_output="A URL to the generated image.",
        agent=image_creator,
        context=[research_task], # Image creator uses researcher's output for the prompt
        # Define a custom callback to extract the prompt from the researcher's output
        # For simplicity, we'll rely on the agent's LLM to extract it, but a parser could be added.
    )

    writing_task = Task(
        description=f"""Using the insights from the research and the generated image URL on '{topic}' for '{platform}',
        develop 3-5 distinct social media post variations.
        Each post should be tailored to the '{platform}'s characteristics (e.g., character limits for Twitter,
        visual focus for Instagram, professional tone for LinkedIn).
        For each post, clearly indicate the generated image URL by embedding it using markdown image syntax.
        Include relevant emojis and at least 3-5 appropriate hashtags per post.
        Ensure variety in the post angles while maintaining engagement.

        IMPORTANT: Format the output as proper markdown with:
        - A main heading for the topic using #
        - Subheadings for each platform (e.g., ## Twitter Posts)
        - Bullet points for each post variation.
        - For each post, include the image URL in the format: `![Alt Text](<IMAGE_URL_HERE>)` directly below the post text.
        - Bold text for emphasis using **text**
        - No code blocks or triple backticks in the output""",
        expected_output=f"""3-5 distinct social media post variations for '{platform}' on '{topic}',
        each with engaging copy, relevant emojis, at least 3-5 hashtags, AND the generated image URL clearly embedded using markdown image syntax.
        Formatted clearly in markdown.""",
        agent=social_media_writer,
        context=[research_task, image_generation_task] # Writer needs both research and image URL
    )
    
    return research_task, image_generation_task, writing_task

def run_social_media_generation(topic, platform):
    """Run the social media content generation crew."""
    research_task, image_generation_task, writing_task = create_social_media_tasks(topic, platform)
    
    crew = Crew(
        agents=[social_media_researcher, image_creator, social_media_writer],
        tasks=[research_task, image_generation_task, writing_task],
        process=Process.sequential,
        verbose=1
    )
    result = crew.kickoff()
    
    return result

# Streamlit App
def main():
    st.title("📱 AI Social Media Content Creator")
    st.markdown("*Powered by Clarifai (Gemini Pro & Imagen/Stable Diffusion), CrewAI & SerperDevTool*")

    st.markdown("""
    This application uses Clarifai and CrewAI to generate engaging social media posts tailored to your chosen platform, **now with AI-generated images powered by Clarifai's image models!**
    
    **How it works:**
    - 🔍 **Social Media Researcher Agent** identifies trends, hashtags, and platform-specific styles, *including visual concepts*.
    - 🎨 **AI Image Creator Agent** generates a relevant image using a Clarifai-hosted image model (e.g., Imagen or Stable Diffusion) based on the researcher's prompt.
    - ✍️ **Social Media Strategist & Writer Agent** crafts compelling, platform-optimized posts, *integrating the generated image*.
    - 🧠 **Powered by** Clarifai's Gemini 2.5 Pro model for text, and Clarifai's image generation models.
    """)

    # Input section
    with st.container():
        topic = st.text_input(
            "Enter the topic for your social media post:",
            placeholder="e.g., The Future of Sustainable Urban Development",
            help="Be specific for better results"
        )
        platform = st.selectbox(
            "Select Target Social Media Platform:",
            options=["Twitter", "LinkedIn", "Instagram", "Facebook"],
            help="Choose the platform to tailor your content"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            generate_button = st.button("✨ Generate Posts & Image", type="primary")

    # Generation section
    if generate_button:
        if not topic.strip():
            st.error("Please enter a topic for the social media post.")
        else:
            with st.spinner(f"🧠 AI agents are crafting posts and images for '{topic}' on {platform}..."):
                try:
                    result = run_social_media_generation(topic, platform)
                    
                    st.success("✅ Social media posts and image generated successfully!")
                    st.markdown("---")
                    
                    # Display the result. The markdown output from the writer should contain the image URL.
                    st.markdown(result)
                    
                    # For a nicer display, you can extract the image URL and use st.image
                    import re
                    image_url_matches = re.findall(r'!\[.*?\]\((https?://\S+)\)', result)
                    if image_url_matches:
                        st.markdown("### Generated Image Preview(s):")
                        # Use set to display unique images if the same URL is repeated
                        for img_url in set(image_url_matches):
                            # Ensure the URL is directly usable for st.image
                            if "api.clarifai.com" in img_url: # Clarifai URLs might need an API key for direct access in some cases, or be public.
                                st.image(img_url, caption="Generated Image (Clarifai)", use_column_width=True)
                            else:
                                st.image(img_url, caption="Generated Image", use_column_width=True)
                    else:
                        st.info("No image URL found in the generated content (or the image generation failed).")

                    # Download option
                    st.download_button(
                        label="📥 Download as Markdown",
                        data=result,
                        file_name=f"{topic.replace(' ', '_').lower()}_{platform.lower()}_posts_with_image.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e) # Display full exception for debugging

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        
        st.markdown("**Environment Variables Required:**")
        st.code("CLARIFAI_PAT=your_clarifai_personal_access_token")
        st.code("SERPER_API_KEY=your_serper_dev_api_key")
        
        st.markdown("**Current Configuration:**")
        st.markdown(f"- **Text LLM:** `Clarifai/Gemini 2.5 Pro`")
        st.markdown(f"- **Search Tool:** `SerperDevTool`")
        st.markdown(f"- **Image Gen Model:** `Clarifai ({STABLE_DIFFUSION_XL_MODEL_URL.split('/')[-1]})`")
        
        st.markdown("**Features:**")
        st.markdown("- Real-time web research for social media trends")
        st.markdown("- **AI-powered image generation via Clarifai models**")
        st.markdown("- AI-powered social media content creation")
        st.markdown("- Platform-specific content generation")
        st.markdown("- Markdown formatted output with embedded images")
        st.markdown("- Download capability")
        
        st.warning("⚠️ Keep your API keys secure and never commit them to version control.")

if __name__ == "__main__":
    main()