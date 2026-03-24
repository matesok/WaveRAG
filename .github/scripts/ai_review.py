import os
import sys
from google import genai

MAX_DIFF_CHARS = 30_000


def load_diff(path: str = 'pr_diff.txt') -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if not content.strip():
            print("Diff file is empty — nothing to review.")
            sys.exit(0)
        if len(content) > MAX_DIFF_CHARS:
            content = content[:MAX_DIFF_CHARS]
            content += "\n\n... [diff truncated — showing first 30 000 characters]"
        return content
    except FileNotFoundError:
        print('Diff file not found. Please ensure the diff is saved to pr_diff.txt')
        sys.exit(1)


def build_prompt(diff: str, pr_title, pr_author) -> str:
    return f"""
        You are a senior software engineer performing a code review.
        Analyze the following code changes and provide:
        - potential bugs
        - readability issues
        - suggestions for improvement
        Code changes:
        {diff.strip()}
        PR Title: {pr_title}
        Author: {pr_author}
        """


def run_ai_review(diff: str, pr_title: str, pr_author: str):
    llm_key = os.getenv('LLM_KEY')
    if not llm_key:
        print("LLM_KEY environment variable not set. Please set it to your language model API key.")
        sys.exit(1)

    client = genai.Client(api_key=llm_key)

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=build_prompt(diff, pr_title, pr_author),
    )

    review_text = response.text
    print("Review generated successfully.")

    return review_text


def save_comment(review: str, pr_number: str) -> None:
    header = (
        f"<!-- ai-review -->\n"
        f"# 🤖 AI Code Review — PR #{pr_number}\n\n"
        f"> Automatyczna analiza wygenerowana przez Claude · "
        f"[Nie zastępuje ludzkiego review]\n\n"
        f"---\n\n"
    )
    footer = (
        "\n\n---\n"
        "_Wygenerowano przez [AI Code Review workflow](.github/workflows/ai-pr-review.yml)_"
    )
    output = header + review + footer

    with open("review_comment.md", "w", encoding="utf-8") as f:
        f.write(output)

    print("💾  Review saved to review_comment.md")


def main() -> None:
    pr_title = os.environ.get("PR_TITLE", "No title")
    pr_author = os.environ.get("PR_AUTHOR", "Unknown")
    pr_number = os.environ.get("PR_NUMBER", "0")

    diff = load_diff("pr_diff.txt")
    review = run_ai_review(diff, pr_title, pr_author)
    save_comment(review, pr_number)


if __name__ == "__main__":
    main()