#!/usr/bin/env python3
"""
Notion to GitHub Sync Agent.

This script:
1. Reads tasks from a Notion database (or specific page)
2. Filters tasks with status "Ready for Claude"
3. Creates GitHub issues with 'claude-task' label
4. Updates Notion task status to "In Progress"

Usage:
- Set up a Notion database with columns: Task, Status, Description, Priority
- Create a Notion integration and share the database with it
- Add secrets to GitHub: NOTION_API_KEY, NOTION_TASK_DATABASE_ID

Database ID for this project: 8fc39f56-17e5-421f-9f3a-d2ae079f806d
"""

import os
from typing import Optional

from notion_client import Client as NotionClient
from github import Github


# Configuration
NOTION_DATABASE_ID = "8fc39f56-17e5-421f-9f3a-d2ae079f806d"


def get_notion_client() -> NotionClient:
    """Initialize Notion client."""
    return NotionClient(auth=os.environ["NOTION_API_KEY"])


def get_github_client() -> Github:
    """Initialize GitHub client."""
    return Github(os.environ["GITHUB_TOKEN"])


def fetch_ready_tasks(notion: NotionClient, database_id: str) -> list[dict]:
    """Fetch tasks with status 'Ready for Claude'."""
    response = notion.databases.query(
        database_id=database_id,
        filter={
            "property": "Status",
            "select": {
                "equals": "Ready for Claude"
            }
        }
    )
    return response.get("results", [])


def fetch_single_page(notion: NotionClient, page_id: str) -> dict:
    """Fetch a single Notion page."""
    return notion.pages.retrieve(page_id=page_id)


def get_page_content(notion: NotionClient, page_id: str) -> str:
    """Get page content as markdown-like text."""
    blocks = notion.blocks.children.list(block_id=page_id)
    content_parts = []
    
    for block in blocks.get("results", []):
        block_type = block.get("type")
        
        if block_type == "paragraph":
            texts = block.get("paragraph", {}).get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in texts)
            content_parts.append(text)
            
        elif block_type == "heading_1":
            texts = block.get("heading_1", {}).get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in texts)
            content_parts.append(f"# {text}")
            
        elif block_type == "heading_2":
            texts = block.get("heading_2", {}).get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in texts)
            content_parts.append(f"## {text}")
            
        elif block_type == "heading_3":
            texts = block.get("heading_3", {}).get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in texts)
            content_parts.append(f"### {text}")
            
        elif block_type == "bulleted_list_item":
            texts = block.get("bulleted_list_item", {}).get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in texts)
            content_parts.append(f"- {text}")
            
        elif block_type == "numbered_list_item":
            texts = block.get("numbered_list_item", {}).get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in texts)
            content_parts.append(f"1. {text}")
            
        elif block_type == "to_do":
            texts = block.get("to_do", {}).get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in texts)
            checked = block.get("to_do", {}).get("checked", False)
            checkbox = "[x]" if checked else "[ ]"
            content_parts.append(f"- {checkbox} {text}")
            
        elif block_type == "code":
            texts = block.get("code", {}).get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in texts)
            language = block.get("code", {}).get("language", "")
            content_parts.append(f"```{language}\n{text}\n```")
    
    return "\n".join(content_parts)


def extract_task_info(page: dict) -> dict:
    """Extract task information from Notion page properties."""
    props = page.get("properties", {})
    
    # Extract title
    title_prop = props.get("Task", {}).get("title", [])
    title = "".join(t.get("plain_text", "") for t in title_prop) if title_prop else "Untitled"
    
    # Extract description
    desc_prop = props.get("Description", {}).get("rich_text", [])
    description = "".join(t.get("plain_text", "") for t in desc_prop) if desc_prop else ""
    
    # Extract priority
    priority_prop = props.get("Priority", {}).get("select")
    priority = priority_prop.get("name") if priority_prop else "Medium"
    
    return {
        "id": page["id"],
        "title": title,
        "description": description,
        "priority": priority,
        "url": page.get("url", ""),
    }


def create_github_issue(gh: Github, task: dict, content: str) -> str:
    """Create a GitHub issue from Notion task."""
    repo = gh.get_repo(os.environ["GITHUB_REPOSITORY"])
    
    # Build issue body
    body = f"""## Task Description
{task['description']}

## Details
{content}

## Notion Link
[View in Notion]({task['url']})

## Priority
{task['priority']}

---
*This issue was automatically created from Notion.*
"""
    
    # Create issue with claude-task label
    issue = repo.create_issue(
        title=task["title"],
        body=body,
        labels=["claude-task"],
    )
    
    return issue.html_url


def update_notion_status(notion: NotionClient, page_id: str, status: str, github_url: str = None):
    """Update Notion task status and GitHub issue link."""
    properties = {
        "Status": {
            "select": {"name": status}
        }
    }
    
    if github_url:
        properties["GitHub Issue"] = {"url": github_url}
    
    notion.pages.update(page_id=page_id, properties=properties)


def main():
    """Main entry point."""
    notion = get_notion_client()
    gh = get_github_client()
    
    database_id = os.environ.get("NOTION_DATABASE_ID", NOTION_DATABASE_ID)
    manual_page_id = os.environ.get("MANUAL_PAGE_ID")
    
    if manual_page_id:
        # Process single page
        print(f"Processing manual page: {manual_page_id}")
        page = fetch_single_page(notion, manual_page_id)
        tasks = [page]
    else:
        # Fetch all ready tasks
        print(f"Fetching tasks from database: {database_id}")
        tasks = fetch_ready_tasks(notion, database_id)
    
    print(f"Found {len(tasks)} task(s) to process")
    
    for page in tasks:
        task = extract_task_info(page)
        print(f"Processing: {task['title']}")
        
        # Get page content
        content = get_page_content(notion, task["id"])
        
        # Create GitHub issue
        issue_url = create_github_issue(gh, task, content)
        print(f"  Created issue: {issue_url}")
        
        # Update Notion status
        update_notion_status(notion, task["id"], "In Progress", issue_url)
        print(f"  Updated Notion status to 'In Progress'")
    
    print("Done!")


if __name__ == "__main__":
    main()
