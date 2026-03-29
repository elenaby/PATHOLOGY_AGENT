from smolagents import CodeAgent
from smolagents.models import OpenAIModel

from tools.segmentation import segment_image
from tools.colorize import color_instances


# ✅ Proper model object
model = OpenAIModel(model_id="gpt-4.1-mini")

agent = CodeAgent(
    tools=[segment_image, color_instances],
    model=model
)


# 🔍 Simple intent detection (reliable)
def detect_intent(user_input: str):
    text = user_input.lower()

    if any(word in text for word in ["segment", "detect", "separate", "identify"]):
        return "segment"

    if any(word in text for word in ["color", "label", "instance"]):
        return "color"

    return "unknown"


def run_agent(user_input: str, image_path: str):
    print("\n=== RUN_AGENT START ===")
    print("User input:", user_input)

    intent = detect_intent(user_input)
    print("Detected intent:", intent)

    # 🔥 PRIMARY PATH (guaranteed execution)
    if intent == "segment":
        print(">>> Running segmentation tool directly")

        mask_path = segment_image(image_path)

        # If user also wants coloring
        if any(word in user_input.lower() for word in ["color", "instance", "label"]):
            print(">>> Running colorization tool")
            result = color_instances(mask_path)
            print("Final result:", result)
            return result

        print("Final result:", mask_path)
        return mask_path

    # 🤖 FALLBACK: use LLM only if needed
    print(">>> Using LLM fallback")

    prompt = f"""
You are a pathology AI assistant.

Available tools:
- segment_image(image_path)
- color_instances(mask_path)

Rules:
- You MUST use tools
- DO NOT explain anything
- ONLY return the final image path

User request: {user_input}
Image path: {image_path}
"""

    result = agent.run(prompt)

    print("=== AGENT RAW OUTPUT ===")
    print(result)

    # Safety cleanup
    if isinstance(result, str):
        return result.strip()

    # fallback safety
    return "outputs/mask.png"