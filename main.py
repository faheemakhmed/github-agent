import os
import sys
import requests
import json
from dotenv import load_dotenv
from portia import Config, DefaultToolRegistry, Portia, PlanRunState, LLMProvider
from portia.cli import CLIExecutionHooks
from pydantic import BaseModel, Field

load_dotenv()

class GitHubPRReviewOutput(BaseModel):
    review_comment: str = Field(
        ..., 
        description="The review comment to be posted to the GitHub PR"
    )

def fetch_github_pr(owner, repo, pr_number):
    """Fetch PR details from GitHub API"""
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}", headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch PR: {response.status_code} - {response.text}")
    
    return response.json()

def fetch_github_pr_diff(owner, repo, pr_number):
    """Fetch PR diff from GitHub API"""
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github.v3.diff"
    }
    
    response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}", headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch PR diff: {response.status_code} - {response.text}")
    
    return response.text

def post_github_pr_review(owner, repo, pr_number, review):
    """Post a review comment to GitHub PR"""
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # First, let's try to post a general review comment (not a line-specific comment)
    data = {
        "body": review
    }
    
    response = requests.post(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments",
        headers=headers,
        json=data
    )
    
    if response.status_code != 201:
        # If that fails, try posting as a pull request review
        # Get the latest commit SHA first
        pr_response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}", headers=headers)
        if pr_response.status_code != 200:
            raise Exception(f"Failed to get PR details for review: {pr_response.status_code}")
        
        pr_data = pr_response.json()
        commit_id = pr_data["head"]["sha"]
        
        # Post as a pull request review
        review_data = {
            "commit_id": commit_id,
            "body": review,
            "event": "COMMENT"
        }
        
        response = requests.post(
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
            headers=headers,
            json=review_data
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to post review: {response.status_code} - {response.text}")
    
    return response.json()

def run_agent():
    """Run the GitHub PR review agent."""
    # Check if Google API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is not set in the environment.")
        print("Please set your Google API key in your .env file.")
        sys.exit(1)
    
    # Check if GitHub token is available
    if not os.getenv("GITHUB_TOKEN"):
        print("Error: GITHUB_TOKEN is not set in the environment.")
        print("Please set your GitHub token in your .env file.")
        sys.exit(1)
    
    try:
        # Create a Portia config with Google GenAI provider and Gemini model
        config = Config.from_default(
            llm_provider=LLMProvider.GOOGLE,
            default_model="google/gemini-1.5-flash-latest",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            default_log_level="DEBUG",
        )
        
        tools = DefaultToolRegistry(config)
        portia = Portia(
            config=config,
            tools=tools,
            execution_hooks=CLIExecutionHooks(),
        )
        
        # Get the GitHub repository and PR number from environment variables or user input
        repo_owner = os.getenv("GITHUB_REPO_OWNER") or input("Enter the repository owner: ")
        repo_name = os.getenv("GITHUB_REPO_NAME") or input("Enter the repository name: ")
        pr_number = os.getenv("GITHUB_PR_NUMBER") or input("Enter the PR number: ")
        
        print("\nFetching PR details...")
        pr_details = fetch_github_pr(repo_owner, repo_name, pr_number)
        
        print("Fetching PR diff...")
        pr_diff = fetch_github_pr_diff(repo_owner, repo_name, pr_number)
        
        print("\nGenerating review...")
        task = (
            f"Analyze the following GitHub pull request:\n\n"
            f"PR Title: {pr_details.get('title', 'N/A')}\n\n"
            f"PR Body: {pr_details.get('body', 'N/A')}\n\n"
            f"Code Diff:\n```\n{pr_diff}\n```\n\n"
            f"Please provide a comprehensive review including:\n"
            f"1. Potential issues or problems\n"
            f"2. Suggestions for improvement\n"
            f"3. Positive feedback on good practices\n"
            f"4. Overall assessment\n\n"
            f"The review should be well-structured and helpful for the PR author."
        )
        
        run = portia.run(
            task,
            structured_output_schema=GitHubPRReviewOutput,
        )
        
        if run.state != PlanRunState.COMPLETE:
            raise Exception(
                f"Plan run failed with state {run.state}. Check logs for details."
            )
        
        review_output = GitHubPRReviewOutput.model_validate(run.outputs.final_output.value)
        review_comment = review_output.review_comment
        
        print("\nGenerated PR review:")
        print(review_comment)
        
        # Human-in-the-loop step
        send_decision = input("\nDo you want to post this review to GitHub? (send/not-send): ")
        
        if send_decision.lower() == "send":
            print("\nPosting review to GitHub...")
            result = post_github_pr_review(repo_owner, repo_name, pr_number, review_comment)
            print("\nReview successfully posted to GitHub!")
            print(f"Review URL: {result.get('html_url', 'N/A')}")
        else:
            print("\nReview was not posted to GitHub.")
        
        return review_output
    
    except Exception as e:
        error_message = str(e)
        if "ResourceExhausted" in error_message or "quota" in error_message:
            print("\n=== GEMINI API QUOTA ERROR ===")
            print("You've exceeded your Gemini API quota. Here are some solutions:")
            print("\n1. Wait for quota reset (free tier quotas reset daily)")
            print("2. Upgrade to a paid tier for higher quotas")
            print("3. Use a different API key from a different Google Cloud project")
            print("\nFor more information on Gemini API quotas:")
            print("- https://ai.google.dev/gemini-api/docs/rate-limits")
        else:
            print(f"\nAn error occurred: {error_message}")
        sys.exit(1)

if __name__ == "__main__":
    run_agent()