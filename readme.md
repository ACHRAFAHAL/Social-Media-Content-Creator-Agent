# 📱 AI Social Media Content Creator

Welcome to the AI Social Media Content Creator! This Streamlit application leverages the power of multi-agent systems with `CrewAI` and advanced AI models from `Clarifai` (for both text generation and image generation) and `SerperDevTool` for real-time web research.

The application allows you to generate engaging and visually appealing social media posts for various platforms based on a given topic, complete with AI-generated images.

## ✨ Features

* **Intelligent Agent Collaboration:** Utilizes three specialized AI agents (`Social Media Researcher`, `AI Image Creator`, `Creative Social Media Strategist & Writer`) working in a sequential workflow.
* **Real-time Web Research:** `SerperDevTool` enables the `Researcher` agent to gather up-to-date information, trends, and relevant hashtags.
* **AI-Powered Text Generation:** Leverages Clarifai's powerful `Gemini 2.5 Pro` model for crafting compelling and platform-optimized social media copy.
* **AI-Powered Image Generation:** Integrates Clarifai's robust image generation models (e.g., Stable Diffusion XL, Imagen 2) to create relevant and high-quality visuals for your posts.
* **Platform-Specific Content:** Tailors post length, tone, and style to chosen social media platforms (Twitter, LinkedIn, Instagram, Facebook).
* **Markdown Output:** Generates clean, ready-to-use markdown output, including embedded image URLs.
* **Download Functionality:** Easily download the generated content as a Markdown file.
* **Streamlit User Interface:** Intuitive and easy-to-use web interface built with Streamlit.

## 🚀 How it Works

The application orchestrates a "crew" of AI agents to achieve the content creation goal:

1.  **🔍 Social Media Researcher Agent:**
    * Conducts comprehensive web research on the given topic.
    * Identifies key facts, trending discussions, popular hashtags, and platform-specific content and *visual* styles.
    * Crucially, it formulates a **detailed, single-sentence image generation prompt** for the `AI Image Creator`.

2.  **🎨 AI Image Creator Agent:**
    * Receives the image prompt from the `Researcher`.
    * Uses a Clarifai-hosted image generation model (e.g., Stable Diffusion XL) to create a visual.
    * Returns the URL of the generated image.

3.  **✍️ Creative Social Media Strategist & Writer Agent:**
    * Takes the research insights from the `Researcher` and the image URL from the `Image Creator`.
    * Crafts 3-5 distinct, engaging, and platform-optimized social media post variations.
    * Embeds the generated image URL directly into the markdown output for each post.

## 🛠️ Setup and Installation

### Prerequisites

* Python 3.9+
* Git (for cloning the repository)

### Environment Variables

You need to set the following environment variables. It's recommended to use a `.env` file for local development.

* `CLARIFAI_PAT`: Your Clarifai Personal Access Token. You can get one from [Clarifai's website](https://www.clarifai.com/settings/security).
* `SERPER_API_KEY`: Your SerperDev API Key for web search. Obtain it from [SerperDev](https://serper.dev/).

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/ai-social-media-creator.git](https://github.com/your-username/ai-social-media-creator.git)
    cd ai-social-media-creator
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```